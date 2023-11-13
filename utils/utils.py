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
