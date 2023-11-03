import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
sys.path.append('..')
import config

def plotLoss(losses, fig_path = 'output/plots/trainingloss.png', title='Loss'):
    # Plotting losses
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(losses, label="Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(fig_path)


def visualizeTransformedData(img, lbl, slice_id):
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


def visualizeSegmentation(img, lbl, name, predicted_label):
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
        plt.savefig('output/plots/segmentation_result.png')


def saveCheckpoint(state, filename='unet_task03_liver.pth'):
    print('Saving model checkpoint..')
    torch.save(state, config.SAVED_MODEL_PATH + filename)
