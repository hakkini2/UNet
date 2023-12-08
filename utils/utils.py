import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
sys.path.append('..')
import config



def calculate_dice_score(y_pred, y):
    """
    https://github.com/ljwztc/CLIP-Driven-Universal-Model/blob/main/utils/utils.py

    y_pred: predicted labels, torch tensor
    y: ground truth labels, torch tensor

    dice = (2*tp)/(2*tp+fp+fn)
    """
    # convert labels to 1 and 0
    # y_pred = torch.where(y_pred > 0.5, 1, 0)

    # convert the tensors to 1D for easier computing
    predict = y_pred.contiguous().view(1, -1)
    target = y.contiguous().view(1, -1)

    # calculate true positives
    tp = torch.sum(torch.mul(predict, target))

    # calculate false negatives
    fn = torch.sum(torch.mul(predict != 1, target))

    # calculate false positives
    fp = torch.sum(torch.mul(predict, target != 1))

    # calculate true negatives
    tn = torch.sum(torch.mul(predict != 1, target != 1))

    # dice = (2*tp)/(2*tp+fp+fn)
    dice = 2 * tp / (torch.sum(predict) + torch.sum(target))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return dice, sensitivity, specificity

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def normalize8(I):
    mn = I.min()
    mx = I.max()

    mx -= mn

    I = ((I - mn)/mx) * 255
    return I.astype(np.uint8)

def plotLoss(losses, fig_path = 'output/plots/trainingloss.png', title='Loss'):
    # Plotting losses
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(losses, label="Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(fig_path)
    plt.close()


def visualizeTransformedData3d(img, lbl, slice_id):
    '''
    img and lbl should have only 3 channels, x,y,z
    '''
    print(f"image shape: {img.shape}, label shape: {lbl.shape}")

    plt.figure("check", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(img[:, :, slice_id], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(lbl[:, :, slice_id])
    plt.savefig('output/plots/visualize_transformed_data.png')
    plt.close()


def visualizeSegmentation3d(img, lbl, name, predicted_label):
    '''
    Visualises a slice of one of the 4 crops of one 3D input volume.
    '''
    with torch.no_grad():

        plt.figure(figsize=(12,4))
        plt.suptitle(name[0], fontsize=14)
        plt.subplot(1,3,1)
        plt.imshow(img[0][0][:,:,60].to('cpu'), cmap='gray')
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(lbl[0][0][:,:,60].to('cpu'), cmap='copper')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(predicted_label[:,:,60], cmap='copper')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'output/plots/segmentation_result_{config.IMG_FORMAT}.png')
        plt.close()


def visualizeSegmentation2d(img, lbl, name, predicted_label):
    with torch.no_grad():

        plt.figure(figsize=(12,4))
        plt.suptitle(name[0], fontsize=14)
        plt.subplot(1,3,1)
        plt.imshow(img[0][0][:,:].to('cpu'), cmap='gray')
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(lbl[0][0][:,:].to('cpu'), cmap='copper')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(predicted_label[0][0][:,:], cmap='copper')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'output/plots/segmentation_result_{config.IMG_FORMAT}.png')
        plt.close()

def saveCheckpoint(state, filename):
    print('Saving model checkpoint..')
    torch.save(state, config.SAVED_MODEL_PATH + filename)


def plot_dice_table(conf, all_dices, output_folder, mode='2D'):
    '''
    mode: ['2D', '3D'] - are we calcualting the dices from 2D slices or 3D avgs
    
    NOTE: all_dices has a different structure depending on the mode:
            2D: {'organ': [('filename', metatensor(dice), ...], 'organ2': [...], ..}
            3D: { organ: {patient_id: avg_dice, ...}, ...}
    '''

    title = "Dice Scores by Organ (2D Slices)" if mode =='2D' else "Dice Scores by Organ (3D Volumes)"
    fig_background_colour = "linen"
    fig_border_colour = "peru"

    # initialize table data
    data = [["Average", "Max", "Min"]]

    # fill in data
    if mode == '2D':
        for organ, dices in all_dices.items():
            dice_values = list(map(lambda dice: dice[1].item(), dices))

            avg = sum(dice_values) / len(dice_values)
            minimum = min(dice_values)
            maximum = max(dice_values)

            data.append([organ, avg, maximum, minimum])
    else:   # 3D
        for organ, dices_by_patient in all_dices.items():
            dice_values = list(dices_by_patient.values())

            avg = sum(dice_values) / len(dice_values)
            minimum = min(dice_values)
            maximum = max(dice_values) 

            data.append([organ, avg, maximum, minimum])          

    column_headers = data.pop(0)
    row_headers = [x.pop(0) for x in data]

    cell_text = []
    for row in data:
        cell_text.append([f"{x:1.4f}" for x in row])

    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

    plt.figure(
        linewidth=2,
        edgecolor=fig_border_colour,
        facecolor=fig_background_colour,
        tight_layout={"pad": 1},
        # figsize=(5,3)
    )

    table = plt.table(
        cellText=cell_text,
        rowLabels=row_headers,
        rowColours=rcolors,
        rowLoc="right",
        colColours=ccolors,
        colLabels=column_headers,
        loc="center",
    )

    # format and plot
    table.scale(1, 1.5)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)
    plt.suptitle(title)
    plt.draw()
    fig = plt.gcf()
    plt.savefig(
        f'{output_folder}/{mode}_dice_score_table.png',
        # bbox='tight',
        edgecolor=fig.get_edgecolor(),
        facecolor=fig.get_facecolor(),
        dpi=150,
    )
    plt.close()
