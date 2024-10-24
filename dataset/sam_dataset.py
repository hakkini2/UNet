import os

import numpy as np
import scipy.ndimage as ndi
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, ScaleIntensityRanged, ToTensord
from scipy import ndimage
from scipy.ndimage import center_of_mass, distance_transform_edt

transforms_sam = Compose(
    [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys="image",
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ToTensord(keys=["image", "label"]),
    ]
)


def get_point_prompt(ground_truth_map):
    """
    ------Individual prompt for each image------
    Creates a point prompt for sam based on the centre of mass
            from the ground truth labels
    Returns: a tuple of the (x,y)-coordinates of the prompt
    """
    cm = center_of_mass(ground_truth_map)
    cm = (round(cm[0]), round(cm[1]))  # round to closest int to get indices

    # print('point prompt: ', cm)

    return cm


def averaged_center_of_mass(loader):
    """
    Calculate just the average of the individual centers of mass
    """
    print("Calculating the averaged center of mass for the point prompt...")
    cms_0 = []
    cms_1 = []
    for i, item in enumerate(loader):
        ground_truth_mask = item["label"].squeeze().to(bool)
        cm = center_of_mass(ground_truth_mask)
        cm_0, cm_1 = round(cm[0]), round(cm[1])
        cms_0.append(cm_0)
        cms_1.append(cm_1)
    avg_0 = sum(cms_0) / len(cms_0)
    avg_1 = sum(cms_1) / len(cms_1)

    print("avg center of mass:", avg_0, avg_1)

    return (avg_0, avg_1)


def compute_center_of_mass_naive(binary_mask):
    """
    Return a list of centers of masses for an image,
    each cluster has an individual cm.

    NOTE: Does not account for the cm being outside
    of the foreground.
    """

    labeled_array, num_features = ndimage.label(binary_mask)
    center_of_mass_list = []

    for label in range(1, num_features + 1):
        # Extract each labeled object
        labeled_object = np.where(labeled_array == label, 1, 0)

        # Compute center of mass for the labeled object
        center_of_mass = ndimage.center_of_mass(labeled_object)

        center_of_mass_list.append(center_of_mass)

    return center_of_mass_list


def compute_center_of_mass(binary_mask):
    """
    Return a list of centers of masses for an image,
    each cluster has an individual cm.

    NOTE: If cm is on top of background, choose
    the closest point belonging to the GT mask.
    """

    labeled_array, num_features = ndimage.label(binary_mask)
    center_of_mass_list = []

    for label in range(1, num_features + 1):
        # Extract each labeled object
        labeled_object = np.where(labeled_array == label, 1, 0)

        # Compute center of mass for the labeled object
        center_of_mass = ndimage.center_of_mass(labeled_object)

        # check if cm is outside of mask, choose closest mask pixel (not the most efficient way)
        cm_y, cm_x = int(center_of_mass[0]), int(center_of_mass[1])

        if labeled_object[cm_y, cm_x] == 0:
            # find all foreground points
            foreground_points = np.column_stack(np.where(labeled_object == 1))

            # compute euclidean distances from the CM to foreground points
            distances = np.sqrt((foreground_points[:, 0] - cm_y) ** 2 + (foreground_points[:, 1] - cm_x) ** 2)

            # find the closest foreground point from the distances
            min_distance_index = np.argmin(distances)
            closest_foreground_point = foreground_points[min_distance_index]
            center_of_mass = tuple(closest_foreground_point)

        center_of_mass_list.append(center_of_mass)

    return center_of_mass_list


def compute_furthest_point_from_edges(binary_mask):
    """
    Return two lists:
    i) the furthest point from the foreground edges in the cluster, and
    ii) the size of the cluster in number of pixels,
    where the size of the lists is the number of clusters.
    """

    labeled_array, num_features = ndimage.label(binary_mask)
    points_list = []
    pixels_list = []

    for label in range(1, num_features + 1):
        # Extract each labeled object
        labeled_object = np.where(labeled_array == label, 1, 0)

        distance_transform = distance_transform_edt(labeled_object)

        furthest = np.unravel_index(np.argmax(distance_transform), distance_transform.shape)
        num_pixels = np.count_nonzero(labeled_object)

        points_list.append(furthest)
        pixels_list.append(num_pixels)

    return points_list, pixels_list


def compute_single_furthest_point_from_edges(binary_mask):
    """
    Return the furthest point from the foreground edges, of the largest cluster.
    """

    labeled_array, num_features = ndimage.label(binary_mask)

    # Find cluster with largest area
    n_pixels = []
    labels = []
    for label in range(1, num_features + 1):
        # Extract each labeled object
        labeled_object = np.where(labeled_array == label, 1, 0)
        n_pixels.append(np.count_nonzero(labeled_object))
        labels.append(label)

    max_label = labels[np.argmax(n_pixels)]

    labeled_object = np.where(labeled_array == max_label, 1, 0)
    distance_transform = distance_transform_edt(labeled_object)

    assert np.count_nonzero(labeled_object) == np.amax(n_pixels)
    furthest = np.unravel_index(np.argmax(distance_transform), distance_transform.shape)

    return furthest, np.amax(n_pixels)


def compute_bounding_boxes(mask_slice, noise=None):
    """
    Returns a list of bounding boxes, one for each cluster.
    """
    labeled_mask, num_features = ndi.label(mask_slice)
    bounding_boxes = []
    for region in range(1, num_features + 1):
        where = np.where(labeled_mask == region)
        x_min, x_max = np.min(where[1]), np.max(where[1])
        y_min, y_max = np.min(where[0]), np.max(where[0])

        # perturbation
        if noise is not None:
            h, w = mask_slice.shape
            perturb_px = noise
            x_min = max(0, x_min - np.random.randint(0, perturb_px))
            x_max = min(w - 1, x_max + np.random.randint(0, perturb_px))
            y_min = max(0, y_min - np.random.randint(0, perturb_px))
            y_max = min(h - 1, y_max + np.random.randint(0, perturb_px))

        bounding_boxes.append([x_min, y_min, x_max, y_max])
    return bounding_boxes


def compute_one_bounding_box(mask_slice):
    where = np.where(mask_slice == 1)
    x_min, x_max = np.min(where[1]), np.max(where[1])
    y_min, y_max = np.min(where[0]), np.max(where[0])

    return [x_min, y_min, x_max, y_max]


def compute_boxes_and_points(mask_slice):
    """
    Computes a bounding box and a point per cluster.
    """
    labeled_mask, num_features = ndi.label(mask_slice)
    bounding_boxes = []
    points = []

    for region in range(1, num_features + 1):
        # bounding box per cluster
        where = np.where(labeled_mask == region)
        x_min, x_max = np.min(where[1]), np.max(where[1])
        y_min, y_max = np.min(where[0]), np.max(where[0])
        bounding_boxes.append([x_min, y_min, x_max, y_max])

        # point per cluster (furthest form foreground edges)
        labeled_object = np.where(labeled_mask == region, 1, 0)
        distance_transform = distance_transform_edt(labeled_object)
        furthest = np.unravel_index(np.argmax(distance_transform), distance_transform.shape)
        points.append(furthest)

    return bounding_boxes, points


def compute_boxes_and_background_points(mask_slice):
    """
    Computes a bounding box and a background point per cluster.
    """
    rows, cols = mask_slice.shape
    radius = 25  # the maximum distance (in pixels) that the point can be from the bounding box edge
    labeled_mask, num_features = ndi.label(mask_slice)
    bounding_boxes = []
    background_points = []

    # loop through clusters
    for region in range(1, num_features + 1):
        # BOX
        where = np.where(labeled_mask == region)
        x_min, x_max = np.min(where[1]), np.max(where[1])
        y_min, y_max = np.min(where[0]), np.max(where[0])
        bounding_boxes.append([x_min, y_min, x_max, y_max])

        # BACKGROUND POINT
        # make a list of all possible coordinates to choose from that are within a range
        candidates = []
        for x in range(x_min - radius, x_max + radius + 1):
            for y in range(y_min - radius, y_max + radius + 1):
                if 0 <= x < cols and 0 <= y < rows:
                    if (x < x_min or x > x_max) or (y < y_min or y > y_max):
                        distance_to_box = min(abs(x - x_min), abs(x - x_max), abs(y - y_min), abs(y - y_max))
                        if distance_to_box <= radius:
                            candidates.append((y, x))

        # probably never happens but
        if not candidates:
            raise ValueError(
                f"No valid background points were found from the image within the given radius of {radius} pixels"
            )

        # choose random point
        background_point = candidates[np.random.randint(len(candidates))]
        background_points.append(background_point)

    return bounding_boxes, background_points


def get_loader(cfg):
    data_dicts = get_data_dicts(cfg)

    dataset = Dataset(data_dicts, transforms_sam)
    loader = DataLoader(dataset, batch_size=cfg.batch_size_sam, num_workers=cfg.num_workers, shuffle=False)

    return loader


def get_data_dicts(cfg):
    """
    Get the images for the given split in a data dict
    """
    organ = cfg.organ.split("_")[1].lower()

    img_dir = os.path.join(cfg.data_path, f"{cfg.split}_2d_images")
    lbl_dir = os.path.join(cfg.data_path, f"{cfg.split}_2d_masks")

    img_fnames = []
    names = []
    for f in os.listdir(img_dir):
        task = f.split("_")[0]

        if task == organ and os.path.isfile(os.path.join(img_dir, f)):
            img_fnames.append(os.path.join(img_dir, f))
            names.append(f.split(".")[0])

    lbl_fnames = []
    for f in os.listdir(lbl_dir):
        task = f.split("_")[0]
        if task == organ and os.path.isfile(os.path.join(lbl_dir, f)):
            lbl_fnames.append(os.path.join(lbl_dir, f))

    data = [
        {"image": img_fname, "label": lbl_fname, "name": name}
        for img_fname, lbl_fname, name in zip(img_fnames, lbl_fnames, names)
    ]

    print(f"{cfg.split} len {format(len(data))}")

    return data
