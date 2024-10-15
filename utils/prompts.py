import numpy as np
import config
from samDataset import (
    get_loader,
    get_point_prompt,
    center_of_mass_from_3d,
    averaged_center_of_mass,
    compute_center_of_mass,
    compute_center_of_mass_naive,
    compute_furthest_point_from_edges,
    compute_bounding_boxes,
    compute_one_bounding_box,
    compute_boxes_and_points,
    compute_boxes_and_background_points
)


def get_point_prompt_prediction(ground_truth_mask, predictor, point_type):
    '''
    Returns SAM's predicted mask using cluster point prompts,
    point prompt type given by point_type.
    '''
    # get list of point prompts - one for each cluster
    if point_type == 'naive_point':
        point_prompts_list = compute_center_of_mass_naive(ground_truth_mask) # naive version of CM
    elif point_type == 'point':
        point_prompts_list = compute_center_of_mass(ground_truth_mask) # CM on background issue fixed
    elif point_type == 'furthest_from_edges_point':
        point_prompts_list, _ = compute_furthest_point_from_edges(ground_truth_mask) # compute the point on foreground that is furthes from the edges

    #initialize mask array 
    mask = np.full(ground_truth_mask.shape, False, dtype=bool)
    # initialize lists for input poitns and labels for plotting
    input_points = []
    input_labels = []

    #loop through centers of mass, get sam's predictions for all and construct the final mask
    for i, point_prompt in enumerate(point_prompts_list):
        #print(f"Center of mass for object {i + 1}: {center_of_mass}")
        input_point = np.array([[round(point_prompt[1]), round(point_prompt[0])]])
        input_label =  np.array([1])
        input_points.append(input_point)
        input_labels.append(input_label)
    
        # get predicted masks
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # CHOOSE THE FIRST MASK FROM MULTIMASK OUTPUT 
        cluster_mask = masks[0]

        # add cluster to final mask
        mask = mask | cluster_mask
    
    return mask, input_points, input_labels



def get_point_prompt_prediction_2(ground_truth_mask, predictor, point_type):
    '''
    Returns SAM's predicted mask using cluster point prompts,
    point prompt type given by point_type.

    NOTE
    Implemented following Meta's own tutorial on using multiple points, but
    does not apparently work as well as the custom approach above.
    '''
    # get list of point prompts - one for each cluster
    if point_type == 'naive_point':
        point_prompts_list = compute_center_of_mass_naive(ground_truth_mask) # naive version of CM
    elif point_type == 'point':
        point_prompts_list = compute_center_of_mass(ground_truth_mask) # CM on background issue fixed
    elif point_type == 'furthest_from_edges_point':
        point_prompts_list, _ = compute_furthest_point_from_edges(ground_truth_mask) # compute the point on foreground that is furthes from the edges
    
    # convert to right format
    input_points = np.array([[round(point[1]), round(point[0])] for point in point_prompts_list])
    input_labels = np.ones(len(point_prompts_list))

    mask, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
    )

    return mask, input_points, input_labels





def get_box_prompt_prediction(ground_truth_mask, predictor):
    '''
    Returns SAM's predicted mask using cluster box prompts.
    '''
    #get a list of bounding boxes - one for each cluster
    box_prompt_list = compute_bounding_boxes(ground_truth_mask, config.USE_NOISE_FOR_BOX_PROMPT)

    #initialize mask array 
    mask = np.full(ground_truth_mask.shape, False, dtype=bool)
    # initialize lists for input boxes for plotting
    input_boxes = []

    #loop through clusters, get sam's predictions for all and construct the final mask
    for i, box_prompt in enumerate(box_prompt_list):
        #print(f'Bounding box for cluster {i+1}: bottom: ({box_prompt[0]}, {box_prompt[1]}), top: ({box_prompt[2]}, {box_prompt[3]})')

        #create input prompt
        input_box = np.array(box_prompt)
        input_boxes.append(input_box)

        # get predicted masks
        cluster_mask, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        #add cluster to final mask
        mask = mask | cluster_mask
    
    return mask, input_boxes




def get_box_with_points_prediction(ground_truth_mask, predictor):
    '''
    Get SAM*s predictions using one large bounding box around all
    the clusters, and one point per cluster.
    '''
    # get cluster point prompts and one boundig box
    point_prompts_list, _ = compute_furthest_point_from_edges(ground_truth_mask)
    box_prompt = compute_one_bounding_box(ground_truth_mask)

    # convert to correct format and create label array to indicate foreground points
    input_points = np.array([[round(point[1]), round(point[0])] for point in point_prompts_list])
    input_labels = np.ones(len(point_prompts_list))
    input_box = np.array(box_prompt)
    
    mask,_,_ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
    )

    # match the format of other box prompts
    input_boxes = [input_box]

    return mask, input_boxes, input_points, input_labels



def get_box_and_point_prompt_prediction(ground_truth_mask, predictor, background_point=False):
    '''
    Get SAM's prediction using a box and a point per cluster
    '''
    if background_point:
        box_prompt_list, point_prompt_list = compute_boxes_and_background_points(ground_truth_mask)
    else:
        box_prompt_list, point_prompt_list = compute_boxes_and_points(ground_truth_mask)

    mask = np.full(ground_truth_mask.shape, False, dtype=bool)
    input_points = []
    input_labels = []
    input_boxes = []

    for i, point_prompt in enumerate(point_prompt_list):
        #point
        input_point = np.array([[round(point_prompt[1]), round(point_prompt[0])]])
        if background_point:
            input_label = np.array([0])
        else:
            input_label =  np.array([1])
        input_points.append(input_point)
        input_labels.append(input_label)

        # box
        input_box = np.array(box_prompt_list[i])
        input_boxes.append(input_box)
    
        # get predicted mask
        cluster_mask, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
        )

        # add cluster mask to final mask
        mask = mask | cluster_mask
    
    return mask, input_boxes, input_points, input_labels


def setminus(mask1, mask2):
  '''
   Returns a new mask with all pixels in mask1 that is not in mask2.
  '''
  return mask1 & ~mask2


def get_box_then_point_prompt_prediction(ground_truth_mask, predictor, point_type='fg'):
    '''
    Returns SAM's predicted mask using cluster box prompts and potentially one interactive point.
    '''
    #get a list of bounding boxes - one for each cluster
    box_prompt_list = compute_bounding_boxes(ground_truth_mask,  
                                             config.USE_NOISE_FOR_BOX_PROMPT)

    #initialize mask array 
    mask = np.full(ground_truth_mask.shape, False, dtype=bool)
    # initialize lists for input boxes for plotting
    input_points = []
    input_labels = []
    input_boxes = []

    # For each cluster, get sam's box prediction, choose a point based on box pred, and do a box+point pred, and construct the final mask by adding all masks together.
    for i, box_prompt in enumerate(box_prompt_list):
       
        #create input prompt
        input_box = np.array(box_prompt)
        input_boxes.append(input_box)

        # get predicted box mask
        box_cluster_mask, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        # Calculate GT mask inside bbox
        bbox_mask = np.full(ground_truth_mask.shape, False, dtype=bool)
        for i in range(ground_truth_mask.shape[0]):
            for j in range(ground_truth_mask.shape[1]):
                # [x_min, y_min, x_max, y_max]
                if j >= input_box[0] and j <= input_box[2] and i >= input_box[1] and i <= input_box[3]:
                    bbox_mask[i,j] = True
        gt_box_mask = ground_truth_mask & bbox_mask

        # Choose which mask to use.
        if point_type == 'fg':
            point_mask = np.squeeze(setminus(gt_box_mask, box_cluster_mask))
            point_prompts_list, pixels_list = compute_furthest_point_from_edges(point_mask)
            label = 1
        elif point_type == 'bg':
            point_mask = np.squeeze(setminus(box_cluster_mask, gt_box_mask))
            point_prompts_list, pixels_list = compute_furthest_point_from_edges(point_mask)
            label = 0
        elif point_type == 'fg/bg':
            fg_point_mask = np.squeeze(setminus(gt_box_mask, box_cluster_mask))
            bg_point_mask = np.squeeze(setminus(box_cluster_mask, gt_box_mask))

            fg_point_prompts_list, fg_pixels_list = compute_furthest_point_from_edges(fg_point_mask)
            bg_point_prompts_list, bg_pixels_list = compute_furthest_point_from_edges(bg_point_mask)
            if len(fg_pixels_list) == 0 and len(bg_pixels_list) == 0:
                mask = mask | box_cluster_mask
                continue
            elif len(fg_pixels_list) == 0 and not len(bg_pixels_list) == 0:
                is_fg_largest = False
            elif not len(fg_pixels_list) == 0 and len(bg_pixels_list) == 0:
                is_fg_largest = True
            else:
                fg_largest, bg_largest = np.amax(fg_pixels_list), np.amax(bg_pixels_list)
                is_fg_largest = fg_largest >= bg_largest

            point_prompts_list = fg_point_prompts_list if is_fg_largest else bg_point_prompts_list
            pixels_list = fg_pixels_list if is_fg_largest else bg_pixels_list
            label = 1 if is_fg_largest else 0
            point_mask = fg_point_mask if is_fg_largest else bg_point_mask
        else:
            assert False
    
        if np.count_nonzero(point_mask) == 0:
            mask = mask | box_cluster_mask
            continue

        largest_cluster_index = np.argmax(pixels_list)
        point_prompt = point_prompts_list[largest_cluster_index]

        input_point = np.array([[round(point_prompt[1]), round(point_prompt[0])]])
        input_label = np.array([label])
        input_points.append(input_point)
        input_labels.append(input_label)
      
        # Create new mask with point and bbox pred
        cluster_mask, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box[None, :],
            multimask_output=False,
        )
      
        #add cluster to final mask
        mask = mask | cluster_mask
    
    return mask, input_boxes, input_points, input_labels