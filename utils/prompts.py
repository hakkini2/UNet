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
        point_prompts_list = compute_furthest_point_from_edges(ground_truth_mask) # compute the point on foreground that is furthes from the edges

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
        point_prompts_list = compute_furthest_point_from_edges(ground_truth_mask) # compute the point on foreground that is furthes from the edges
    
    # convert to right format
    input_points = np.array([[round(point[1]), round(point[0])] for point in point_prompts_list])
    input_labels = np.ones(len(point_prompts_list))

    mask, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
    )

    return mask, input_points, input_labels





def get_box_prompt_prediction(ground_truth_mask, predictor, use_noise=False):
    '''
    Returns SAM's predicted mask using cluster box prompts.
    '''
    #get a list of bounding boxes - one for each cluster
    box_prompt_list = compute_bounding_boxes(ground_truth_mask, noise=use_noise)

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
    point_prompts_list = compute_furthest_point_from_edges(ground_truth_mask)
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
