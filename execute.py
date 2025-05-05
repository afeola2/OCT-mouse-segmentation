import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.fcn import FCNHead

from torch.utils.data import DataLoader

from utils import *
from plot_utils import *
from customloader import SegmentationDataset, InferenceDataset

import copy 

from matplotlib.backends.backend_pdf import PdfPages

from train_utils2 import *
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def main():

    parser = argparse.ArgumentParser(description="Mouse images segmentation.")
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    parser.add_argument('--image_dir', type=str,required=False, help='Image directory.')

    parser.add_argument('--save_dir', type=str, default=True, required=True, help='Save results in a folder')

    parser.add_argument('--save_images', type=str, default=True, required=False, help='Save images in a folder')

    parser.add_argument('--num_images', type=int,required=False, help='Cap images if too many.')

    args = parser.parse_args()

    
    if args.num_images: 
        tiff_files = [os.path.join(root, file) for root, dirs, files in os.walk(args.image_dir) if 'Image' in root for file in files if file.endswith('.tiff')][:args.num_images]
        sort_nicely(tiff_files)

    else:
        tiff_files = [os.path.join(root, file) for root, dirs, files in os.walk(args.image_dir) if 'Image' in root for file in files if file.endswith('.tiff')]
        sort_nicely(tiff_files)


    print(f'Retrieved {len(tiff_files)} images. Analyzing...')
    
    #  Images in loader
    dataset = InferenceDataset(tiff_files)
    inference_dataloader = DataLoader(dataset, batch_size=1, shuffle=False,drop_last=False)

    # Load model 
    model_path = load_latest_model('./weights',best=False, prefix='Finetuning')
    model = models.segmentation.deeplabv3_resnet50(weights=None)
    model.classifier = DeepLabHead(2048, num_classes=4)

    model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    model.to(device)
    
    model.eval()
    output_images=[]
    output_names=[]
    input_images=[]
    with torch.no_grad():
        for batch in inference_dataloader:
            images, name = batch[0], batch[1][0]
            #print(image.shape)
            images= images.to(device)
            
            # Get outputs from two models
            outputs1 = model(images)['out'].argmax(1).detach()
            
            output_images.append(outputs1.squeeze(0).cpu().numpy())
            input_images.append(images.squeeze(0).cpu().numpy())
            

            # Extract filename from the path
            filename = name.split("/")[-1]  

            # Extract part after "image_"
            after_image = filename.split("image_")[-1]
            # Remove file extension
            after_image_no_ext = after_image.split(".")[0]

            output_names.append(after_image_no_ext)

    #Thicness calculation
    thickness_dict={}

    thickness_dict['names']=[]

    thickness_dict['names']=output_names

    thickness_dict=thickness(output_images,thickness_dict, ['Average thickness'])

    thickness_dict=thickness(output_images,thickness_dict, ['Average thickness Q1'], Q='Q1')
    thickness_dict=thickness(output_images,thickness_dict, ['Average thickness Q2'], Q='Q2')
    thickness_dict=thickness(output_images,thickness_dict, ['Average thickness Q3'], Q='Q3')
    thickness_dict=thickness(output_images,thickness_dict, ['Average thickness Q4'], Q='Q4')

    save_thickness_to_excel(thickness_dict, filename=args.save_dir+"/thickness_results.xlsx")

    if args.save_images:
        save_images_with_color(output_images, output_names, save_path=args.save_dir+'/segmentations', ext='Mask')
        save_images(input_images, output_names, args.save_dir+'/images')


    

    


if __name__ == "__main__":
    main()