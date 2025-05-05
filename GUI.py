import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torch.utils.data import DataLoader
import os
import tkinter as tk
from tkinter import filedialog, messagebox

from utils import *
from plot_utils import *
from customloader import SegmentationDataset, InferenceDataset
from train_utils2 import *
import os
import sys

# Function to select a folder
def select_folder(entry_widget):
    folder_selected = filedialog.askdirectory(title="Select Folder")
    if folder_selected:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, folder_selected)


# Function to run the segmentation
def run_segmentation():
    # Ensure script runs from correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)


    image_dir = image_dir_entry.get().strip()
    save_dir = save_dir_entry.get().strip()
    num_images = num_images_entry.get().strip()

    if not image_dir or not save_dir:
        messagebox.showerror("Error", "Please select both image and save directories.")
        return

    try:
        num_images = int(num_images) if num_images else None
    except ValueError:
        messagebox.showerror("Error", "Number of images must be an integer.")
        return

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Retrieve TIFF files
    if num_images:
        tiff_files = [os.path.join(root, file) for root, _, files in os.walk(image_dir) if "Image" in root for file in files if file.endswith(".tiff")][:num_images]
    else:
        tiff_files = [os.path.join(root, file) for root, _, files in os.walk(image_dir) if "Image" in root for file in files if file.endswith(".tiff")]
       # print(tiff_files)
    if not tiff_files:
        messagebox.showerror("Error", "No TIFF images found in the selected directory.")
        return

    sort_nicely(tiff_files)
    status_label.config(text=f"Processing {len(tiff_files)} images...")

    # Load dataset
    dataset = InferenceDataset(tiff_files)
    inference_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    # Load model
    model_path = load_latest_model('./weights', best=False, prefix="Finetuning")
    model = models.segmentation.deeplabv3_resnet50(weights=None)
    model.classifier = DeepLabHead(2048, num_classes=4)

    model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    model.to(device)
    model.eval()

    output_images, output_names, input_images = [], [], []
    
    with torch.no_grad():
        for batch in inference_dataloader:
            images, name = batch[0], batch[1][0]
            images = images.to(device)
          #  print(name)
           # print(images.shape)
            
            outputs1 = model(images)["out"].argmax(1).detach()
            
            output_images.append(outputs1.squeeze(0).cpu().numpy())
            input_images.append(images.squeeze(0).cpu().numpy())

            filename = name.split("/")[-1]
            after_image_no_ext = filename.split("image_")[-1].split(".")[0]

            output_names.append(after_image_no_ext)

    # Thickness calculation
    thickness_dict = {"names": output_names}
    thickness_dict = thickness(output_images, thickness_dict, ["Average thickness"])
    thickness_dict = thickness(output_images, thickness_dict, ["Average thickness Q1"], Q="Q1")
    thickness_dict = thickness(output_images, thickness_dict, ["Average thickness Q2"], Q="Q2")
    thickness_dict = thickness(output_images, thickness_dict, ["Average thickness Q3"], Q="Q3")
    thickness_dict = thickness(output_images, thickness_dict, ["Average thickness Q4"], Q="Q4")

    save_thickness_to_excel(thickness_dict, filename=os.path.join(save_dir, "thickness_results.xlsx"))

    # Save images
    save_images_with_color(output_images, output_names, save_path=os.path.join(save_dir, "segmentations"), ext="Mask")
    save_images(input_images, output_names, os.path.join(save_dir, "images"))

    status_label.config(text="Processing complete! Results saved.")
    messagebox.showinfo("Success", "Segmentation completed successfully!")


# GUI Setup
root = tk.Tk()
root.title("Mouse Retinal Segmentation")
root.geometry("500x350")

# Image Directory Selection
tk.Label(root, text="Image Directory:").pack(pady=5)
image_dir_entry = tk.Entry(root, width=50)
image_dir_entry.pack()
tk.Button(root, text="Browse", command=lambda: select_folder(image_dir_entry)).pack()

# Save Directory Selection
tk.Label(root, text="Save Directory:").pack(pady=5)
save_dir_entry = tk.Entry(root, width=50)
save_dir_entry.pack()
tk.Button(root, text="Browse", command=lambda: select_folder(save_dir_entry)).pack()

# Number of Images
tk.Label(root, text="Number of Images (Optional):").pack(pady=5)
num_images_entry = tk.Entry(root, width=20)
num_images_entry.pack()

# Run Button
tk.Button(root, text="Run Segmentation", command=run_segmentation, fg="white", bg="blue").pack(pady=20)

# Status Label
status_label = tk.Label(root, text="Select directories and click 'Run Segmentation'", fg="green")
status_label.pack()

root.mainloop()
