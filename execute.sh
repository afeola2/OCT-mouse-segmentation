IMAGE_DIR='/home/feolalab/Desktop/Analyzed mouse retinal scans'
SAVE_DIR="./testOCT"
NUM_IMAGES=5
SAVE_IMAGES=True


python execute.py --image_dir "$IMAGE_DIR"  --save_dir "$SAVE_DIR"    --num_images "$NUM_IMAGES"


#Resize images when saving as tiffs first 