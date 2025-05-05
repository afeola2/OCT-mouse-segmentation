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
import gc
import torch

import glob


def load_latest_model(directory_path,best=False, prefix='exp1_model1', suffix='.pth'):
    """
    Load the latest model file name that matches the given prefix and suffix from the specified directory.

  
    """
    # Define the pattern to search for files
    pattern = f'{prefix}*{suffix}'
    search_pattern = os.path.join(directory_path, pattern)

    # Find all files matching the pattern
    matching_files = glob.glob(search_pattern)

    # Ensure at least one file matches the pattern
    if matching_files:
        # Sort files to get the latest (assuming filenames contain sortable epoch numbers)
        matching_files.sort()
        # Get the last file (or choose accordingly)
        if best:
            latest_file = matching_files[-2]
        else:
            latest_file = matching_files[-1]

        return latest_file

    
#Count folders
def count_and_list_folders(directory):
    folder_names = [item.path for item in os.scandir(directory) if item.is_dir()]
    folder_count = len(folder_names)
    return folder_count, folder_names


#Check every folder has Images and masks subfolder
def check_subfolders(folder_list, subfolder_name):
    newlist=[]
    for folder in folder_list:
        subfolder_path = os.path.join(folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            newlist.append(folder)
    return newlist




#https://stackoverflow.com/questions/4623446/how-do-you-sort-files-numerically
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)



def get_files(root_folder, extension):
    tiff_files = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(extension):
                tiff_files.append(os.path.abspath(os.path.join(subdir, file)))
    return tiff_files



def remove_bad_labels(mask_list_c, mask_list_tt, image_list):
    for i in reversed(range(len(mask_list_c))):  # Iterate in reverse to avoid index issues when popping
        # Load the masks
        mask_choroid = np.asarray(cv2.imread(mask_list_c[i], cv2.IMREAD_UNCHANGED)).astype(np.int64)[1536:]
        masks_C = np.where(mask_choroid != 0, 1, 0)
        
        print(np.unique(masks_C))

        mask_TT = np.asarray(cv2.imread(mask_list_tt[i], cv2.IMREAD_UNCHANGED)).astype(np.int64)[1536:]
        masks_TT = np.where(mask_TT != 0, 2, 0)
        print(np.unique(masks_TT))

        # Combine the masks
        mask = masks_C + masks_TT
        print(np.unique(mask))

        # Check if the unique values in the combined mask are not exactly 3
        if len(np.unique(mask)) != 3:
            print(f'Index: {i}, Paths: {mask_list_c[i]}, {mask_list_tt[i]}')
            #print(np.unique(mask))
            mask_list_c.pop(i)
            mask_list_tt.pop(i)
            image_list.pop(i)
    
    return mask_list_c, mask_list_tt, image_list


def dice_coefficient(pred, target, num_classes=9, epsilon=1e-6):
    """Compute the Dice coefficient for each class."""
    dice = torch.zeros(num_classes, device=pred.device)
    pred = pred.argmax(dim=1)  # Get the class with the highest score for each pixel

    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c

        intersection = (pred_c & target_c).sum().float()
        pred_c_sum = pred_c.sum().float()
        target_c_sum = target_c.sum().float()

        dice_c = (2. * intersection + epsilon) / (pred_c_sum + target_c_sum + epsilon)
        dice[c] = dice_c
    
    return dice

    

def shuffle_and_divide_indexes(total_files, seed):
    random.seed(seed)
    indexes = list(range(total_files))
    random.shuffle(indexes)
    
    size_80 = int(0.8 * total_files)
    size_10 = (total_files - size_80) // 2
    
    indexes_80 = indexes[:size_80]
    indexes_10_a = indexes[size_80:size_80 + size_10]
    indexes_10_b = indexes[size_80 + size_10:]
    
    return indexes_80, indexes_10_a, indexes_10_b

def reorder_list(images, masks,masks2, indexes):
    images=[images[i] for i in indexes]
    masks=[masks[i] for i in indexes]
    masks2=[masks2[i] for i in indexes]
    return images, masks , masks2






def save_lists_and_dicts_to_excel(train_lists, test_dicts,twomodels=True, filename='output.xlsx'):
    # Create a DataFrame for training lists

    
    if twomodels:
        df_train = pd.DataFrame({
            'Combined Loss': train_lists[0],
            'Loss model 1': train_lists[1],
            'Loss model 2': train_lists[2]
        })
    else: 
        df_train = pd.DataFrame({
            'Loss model 1': train_lists[0]
        })
        
    dice_sheet_names=['Dice 1', 'Dice 2', 'Dice comb']
    # Create a Pandas Excel writer using openpyxl as the engine
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Write the training DataFrame to a sheet
        df_train.to_excel(writer, sheet_name='Train', index=False)
        
        # Write each test class dictionary to different sheets
        for dict_index, test_dict in enumerate(test_dicts):
            df_test = pd.DataFrame(test_dict)
            df_test.columns = [f'Dice_class_{c}' for c in df_test.columns]
            df_test.to_excel(writer, sheet_name=dice_sheet_names[dict_index], index=False)





def delete_torch_models():
    # Get the global variables
    global_vars = globals()
    
    # List to hold names of all detected models
    model_names = []

    # Iterate over all items in the global namespace
    for name, obj in global_vars.items():
        # Check if the object is an instance of torch.nn.Module
        if isinstance(obj, torch.nn.Module):
            model_names.append(name)

    # Delete the models
    for name in model_names:
        print(f"Deleting model: {name}")
        del global_vars[name]

    # Collect garbage to remove deleted models from memory
    gc.collect()

    # Clear GPU memory cache if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Deleted {len(model_names)} models from memory")


def thickness(output_images, thickness_dict, col_arguments, Q=None):

    # Ensure col_arguments is a list
    if not isinstance(col_arguments, list):
        col_arguments = [col_arguments]  

    thickness_dict[col_arguments[0] + ' Retina']=[]
    thickness_dict[col_arguments[0]  + ' Choroid']=[]
    thickness_dict[col_arguments[0]  + ' Sclera']=[]
    thickness_dict[col_arguments[0]  + ' Total']=[]

    # Define lookup dictionary for Q2, Q3, Q4
    Q_mapping = {
        'Q2': (125, 375),
        'Q3': (375, 525),
        'Q4': (525, 875)
    }

    # Process images
    for image in output_images:
        if Q == 'Q1':
            # Q1: Two disjoint regions (0-125 and 875-1000)
            cropped_image = np.concatenate((image[:, 0:125], image[:, 875:1000]), axis=1)
        elif Q in Q_mapping:
            # Other quadrants (Q2, Q3, Q4): Single contiguous region
            idx1, idx2 = Q_mapping[Q]
            cropped_image = image[:, idx1:idx2]
        else:
            # No cropping (either Q is None or not recognized)
            cropped_image = image  


        # Compute thickness per layer
        resolution=5.2471/2048
        #total_mask = np.ma.masked_array(cropped_image, cropped_image > 0)
        #total_mask = np.ma.masked_array(cropped_image, cropped_image == 0)
        total_mask = cropped_image > 0  # shape: (height, width)
        total_thickness = np.mean(np.sum(total_mask, axis=0)*resolution)
        thickness_dict[col_arguments[0] + ' Total'].append(total_thickness)

        # Compute thickness per layer
        for layer in range(1, 4):
            layer_mask = np.ma.masked_array(cropped_image, cropped_image == layer)
            layer_thickness = np.mean(np.sum(layer_mask.mask, axis=0))

            # Assign thickness to the correct list
            if layer == 1:
                thickness_dict[col_arguments[0] + ' Retina'].append(layer_thickness)
            elif layer == 2:
                thickness_dict[col_arguments[0] + ' Choroid'].append(layer_thickness)
            elif layer == 3:
                thickness_dict[col_arguments[0] + ' Sclera'].append(layer_thickness)

    return thickness_dict


def save_thickness_to_excel(thickness_dict, filename="thickness_results.xlsx"):
    """
    Saves the thickness dictionary to an Excel file.
    
    Parameters:
        thickness_dict (dict): Dictionary where keys are column names and values are lists.
        filename (str): Name of the Excel file to save.
    """
    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame.from_dict(thickness_dict, orient='columns')

    # Save DataFrame to Excel
    df.to_excel(filename, index=False)

    print(f"Saved thickness results to {filename}")

# Example usage
#save_thickness_to_excel(thickness_dict)
