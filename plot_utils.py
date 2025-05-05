
import os
import re
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import cv2
import torch 
from matplotlib import pyplot as plt
import random
import pandas as pd 
import seaborn as sns




def plot_dice_scores(dice_scores, dice_scores_val=None, title='Dice Coefficient by Class over Epochs'):
    plt.figure(figsize=(14, 8))

    epochs = range(1, len(next(iter(dice_scores.values()))) + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(dice_scores)))  # Get a set of colors

    # Plot training dice scores
    for i, (class_id, scores) in enumerate(dice_scores.items()):
        color = colors[i % len(colors)]
        plt.plot(epochs, scores, linestyle='-', color=color, label=f'Class {class_id} (Train)')

    # Plot validation dice scores if provided
    if dice_scores_val:
        for i, (class_id, scores) in enumerate(dice_scores_val.items()):
            color = colors[i % len(colors)]
            plt.plot(epochs, scores, linestyle='--', marker='x', color=color, label=f'Class {class_id} (Val)')

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Dice Coefficient', fontsize=14)
    plt.title(title, fontsize=16)

    # Create custom legend
    handles = [plt.Line2D([0], [0], color='k', linestyle='-', label='Train'),
               plt.Line2D([0], [0], color='k', linestyle='--', marker='x', label='Val')]
    for i, class_id in enumerate(dice_scores.keys()):
        handles.append(plt.Line2D([0], [0], color=colors[i % len(colors)], lw=2, label=f'Class {class_id}'))

    plt.legend(handles=handles, fontsize=12, title='Legend')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Example Usage
# plot_dice_scores(dice_scores_train, dice_scores_val)


def plot_comparison(image, labels, output1, output2, index, combined=False, alpha=0, beta=0):
    """Plot the image, labels, and two sets of outputs for a specific index."""
    # Convert tensors to numpy arrays for plotting
    image_np = image[index].permute(1, 2, 0).cpu().numpy()  # HWC format
    labels_np = labels[index].cpu().numpy()
    output1_np = output1[index].argmax(0).cpu().numpy()  # Get the predicted class for each pixel
    output2_np = output2[index].argmax(0).cpu().numpy()  # Get the predicted class for each pixel

    if combined:
        n=5
        combined_outputs= (alpha*output1[index]+beta*output2[index]).argmax(0).cpu().numpy()

    else:
        n=4
        
    # Create a figure with subplots
    fig, axes = plt.subplots(1, n, figsize=(20, 5))
    print(image_np.shape)
    # Plot the input image
    axes[0].imshow(image_np[:,:,0], cmap='gray')
    axes[0].set_title('Image')
    axes[0].axis('off')

    # Plot the ground truth labels
    axes[1].imshow(labels_np)
    axes[1].set_title('Labels')
    axes[1].axis('off')

    # Plot the first set of outputs
    axes[2].imshow(output1_np)
    axes[2].set_title('Output 1')
    axes[2].axis('off')

    # Plot the second set of outputs
    axes[3].imshow(output2_np)
    axes[3].set_title('Output 2')
    axes[3].axis('off')


    if combined:
            # Plot the second set of outputs
        axes[4].imshow(combined_outputs)
        axes[4].set_title('Combined outputs')
        axes[4].axis('off')

    # Display the plots
    plt.show()



# Modify plot_finetune to return the figure
def plot_finetune(image, labels, output1=None, index=0, show=True):
    """Plot the image, labels, and outputs for a specific index."""
    # Convert tensors to numpy arrays for plotting
    image_np = image[index].permute(1, 2, 0).cpu().numpy()  # HWC format
    labels_np = labels[index].cpu().numpy()
    output1_np = output1[index].argmax(0).cpu().numpy() if output1 is not None else None # Get the predicted class for each pixel

    # Create a figure with subplots
    num_plots = 2 if output1 is None else 3
    fig, axes = plt.subplots(1, num_plots, figsize=(20, 5))

    # Plot the input image
    axes[0].imshow(image_np[:, :, 0], cmap='gray')
    axes[0].set_title('Image')
    axes[0].axis('off')

    # Plot the ground truth labels
    axes[1].imshow(labels_np)
    axes[1].set_title('Labels')
    axes[1].axis('off')

    if output1 is not None:
        # Plot the output predictions
        axes[2].imshow(output1_np)
        axes[2].set_title('Network Prediction')
        axes[2].axis('off')
    
    # Show or return figure based on `show`
    if show:
        plt.show()
    else:
        return fig



# Define a wrapper function to capture figures from `plot_finetune`
def capture_plot_finetune(images, labels, outputs1, index):
    # Create a new figure
    fig, ax = plt.subplots()
    
    # Assuming plot_finetune takes an axis or figure as an argument (if not, 
    # it might need to be modified to accept ax or fig parameters)
    plot_finetune(images, labels, outputs1, index=index, show=False)
    
    # Return the figure to save it later
    return fig


def plot_dict(epoch_loss, modes, phases, nplots=2):
    if nplots==2:
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot losses
        for i, phase in enumerate(phases):
            for mode in modes:
                axes[i].plot(epoch_loss[phase][mode], label=f'{mode.capitalize()} Loss')
            axes[i].set_title(f'{phase.capitalize()} Losses')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].legend()
    else: 
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        # Plot losses
        for i, mode in enumerate(modes):
            for phase in phases:
                axes[i].plot(epoch_loss[phase][mode], label=f'{phase.capitalize()} Loss')
            axes[i].set_title(f'{mode.capitalize()} Losses')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].legend()
    # Show plots
    plt.tight_layout()
    #plt.title('Experiment 1')
    plt.show()


def plot_losses_3_panels(epoch_loss, modes=['model1']): #Plot losses per mode (3 panels)
    '''Plot losses in mode panels. This function plots train and val loss in one plot. Panels depend on modes provided'''
    # Set the style and context for the plots
    sns.set(style="whitegrid", context="talk")

    # Extract the phases
    phases = ['train', 'val']

    # Create subplots
    fig, axes = plt.subplots(1, len(modes), figsize=(12, 4))

    # Handle the case when there's only one mode
    if len(modes) == 1:
        axes = [axes]

    # Define colors for training and validation phases
    colors = {'train': 'tab:blue', 'val': 'tab:orange'}

    # Plot losses for each mode
    for i, mode in enumerate(modes):
        for phase in phases:
            axes[i].plot(epoch_loss[phase][mode], label=f'{phase.capitalize()} Loss', color=colors[phase], linewidth=2)
        axes[i].set_title(f'{mode.capitalize()} Losses', fontsize=18)
        axes[i].set_xlabel('Epoch', fontsize=14)
        axes[i].set_ylabel('Loss', fontsize=14)
        axes[i].legend(fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].tick_params(axis='both', which='major', labelsize=12)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the plot as a high-resolution image
    # plt.savefig('model_losses_separated.png', dpi=300)

    # Show plots
    plt.show()

def plot_losses_2_panels(epoch_loss, phases=['train', 'val'], modes=['model1']):
    '''Plots 2 panels, train vs validation. Lines in panel depend on provided modes'''
    # Set the style and context for the plots
    sns.set(style="whitegrid", context="talk")

    # Create subplots
    fig, axes = plt.subplots(1, len(phases), figsize=(12, 4))

    # Handle the case when there's only one phase
    if len(phases) == 1:
        axes = [axes]

    # Plot losses
    for i, phase in enumerate(phases):
        for mode in modes:
            axes[i].plot(epoch_loss[phase][mode], label=f'{mode.capitalize()} Loss')
        axes[i].set_title(f'{phase.capitalize()} Losses')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show plots
    plt.show()


def dice_scores_to_df(dice_scores):
    '''Function to prepare dice dict (previously saved as pickle) to be plotted by seaborn'''
    rows = []
    for phase in ['train', 'val']:
        for model in dice_scores[phase]:
            for cls in dice_scores[phase][model]:
                for epoch, score in enumerate(dice_scores[phase][model][cls], 1):
                    rows.append([epoch, phase, model, cls, score])
    return pd.DataFrame(rows, columns=['Epoch', 'Phase', 'Model', 'Class', 'Dice'])

# Function to plot the Dice coefficients
def plot_dice_coefficients_2_panels(dice_scores, name=None):
    '''Function to plot train vs val dice coefficients. Each model is plotted as a different type of line'''
    df = dice_scores_to_df(dice_scores)
    phases = df['Phase'].unique()
    models = df['Model'].unique()
    classes = df['Class'].unique()
    
    # Define a color palette
    palette = sns.color_palette("tab10", len(classes))
    
    # Define line styles for each model
    line_styles = ['-', '--', ':']
    
    fig, axes = plt.subplots(1, len(phases), figsize=(20, 6), sharey=True)
    
    for ax, phase in zip(axes, phases):
        df_phase = df[df['Phase'] == phase]
        
        for cls, color in zip(classes, palette):
            for model, line_style in zip(models, line_styles):
                df_class_model = df_phase[(df_phase['Class'] == cls) & (df_phase['Model'] == model)]
                sns.lineplot(x='Epoch', y='Dice', data=df_class_model, ax=ax, color=color, linestyle=line_style)
        
        ax.set_title(f'{phase.capitalize()} Dice Coefficients')
        ax.set_xlabel('Epoch')
        if ax == axes[0]:
            ax.set_ylabel('Dice Coefficient')
        ax.grid(True)

    # Create a custom legend
    legend_elements = [plt.Line2D([0], [0], color=color, label=f'Class {cls}') for cls, color in zip(classes, palette)]
    for model, line_style in zip(models, line_styles):
        legend_elements.append(plt.Line2D([0], [0], color='black', linestyle=line_style, label=model.capitalize()))
    
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(classes) + len(models), title='Classes and Models', bbox_to_anchor=(0.5, 1.15))
    
    fig.suptitle(f'Dice Coefficients for Different Models and Phases. {name}', fontsize=30)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_dice_coefficients_3_panels(dice_scores, name=None):
    df = dice_scores_to_df(dice_scores)
    models = df['Model'].unique()
    classes = df['Class'].unique()
    
    # Define a color palette
    palette = sns.color_palette("tab10", len(classes))
    
    # Determine number of subplots needed
    n_plots = len(models)
    
    fig, axes = plt.subplots(1, n_plots, figsize=(20, 6), sharey=True)
    
    # Handle the case when there's only one model
    if n_plots == 1:
        axes = [axes]
    
    for ax, model in zip(axes, models):
        df_model = df[df['Model'] == model]
        
        for cls, color in zip(classes, palette):
            df_class_train = df_model[(df_model['Class'] == cls) & (df_model['Phase'] == 'train')]
            df_class_val = df_model[(df_model['Class'] == cls) & (df_model['Phase'] == 'val')]
            
            sns.lineplot(x='Epoch', y='Dice', data=df_class_train, ax=ax, color=color, linestyle='-')
            sns.lineplot(x='Epoch', y='Dice', data=df_class_val, ax=ax, color=color, linestyle='--')
        
        ax.set_title(f'Dice Coefficient for {model}')
        ax.set_xlabel('Epoch')
        if ax == axes[0]:
            ax.set_ylabel('Dice Coefficient')
        ax.grid(True)

    # Create a custom legend
    legend_elements = [plt.Line2D([0], [0], color=color, label=f'Class {cls}') for cls, color in zip(classes, palette)]
    legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='-', label='Train'))
    legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', label='Val'))
    
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(classes) + 2, title='Classes and Phases', bbox_to_anchor=(0.5, 1.15))
    
    fig.suptitle(f'Dice Coefficients for Different Models and Phases. {name}', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



def plot_alpha(alpha, num_epochs):
    plt.plot(range(num_epochs),alpha)
    plt.plot(range(num_epochs),[1-item for item in alpha])

    plt.title('Alpha and Beta')
    plt.xlabel('Epochs')
    plt.ylabel('Alpha values')
    plt.legend(['Alpha', 'Beta'])
    plt.show()

def save_images_with_color(plist, nlist, save_path, ext='Mask'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f'makss created path: {save_path}')
    # Define a colormap: you can customize this as needed
    colormap = {
    0: [0, 0, 0],         # Black (background)
    1: [128, 0, 0],       # Navy blue (darkest)
    2: [255, 255, 255],       # Pure blue
    3: [0, 0, 250],   # Light sky blue (lightest)
    }


    for name, prediction in zip(nlist, plist):
        print(f'name: {name}')
        # Ensure the prediction array has the right shape
        prediction = np.squeeze(prediction)
       # print(prediction.shape)
        
        # Create an RGB image using the colormap
        height, width = prediction.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        for value, color in colormap.items():
            rgb_image[prediction == value] = color
        
        # Convert the array to an image and save it
        image = Image.fromarray(rgb_image)
        #print(os.path.join(save_path, ext+ name + '.tiff'))
       # file_name=[ext+'_'+name+'.tiff']
        #print(f'file_name: {ext+'_'+name+'.tiff'}')
        image.save(os.path.join(save_path,ext+'_'+name+'.tiff'))
    



def save_images(tiffs, list_of_files, save_path, ext='Image'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for name, image in zip(list_of_files, tiffs):
        if image.ndim == 3 and image.shape[0] < 10:  # If (C, H, W), change to (H, W, C)
            image = np.moveaxis(image, 0, -1)

        # Convert to 8-bit grayscale
        img_8bit = normalize_image(image[..., 0])  # Use only first channel

        pil_image = Image.fromarray(img_8bit)
        save_name = f"{ext}_{name}.tiff"
        pil_image.save(os.path.join(save_path, save_name))

def normalize_image(image):
    """Normalize image to [0,255] for uint8 conversion"""
    image = image.astype(np.float32)  # Ensure float type
    image_min, image_max = image.min(), image.max()
    
    # Avoid division by zero if the image is uniform
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min) * 255
    return image.astype(np.uint8)
