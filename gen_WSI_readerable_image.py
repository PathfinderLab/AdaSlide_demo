import os
import re
import glob
import pyvips
import parmap
import argparse
import openslide
from tqdm.auto import tqdm

# Function to extract coordinates from filename
def extract_coords(filename, is_downsample=False):
    base = filename.split("_")[-1].split(".")[0]
    x, y = base.split("-")
    x, y = int(x), int(y)
    
    if is_downsample == True:
        x = int(x / 2)
        y = int(y / 2)
    
    return x, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project')
    parser.add_argument('--lambda_cond', default="lambda_000")
    parser.add_argument('--patch_format', default="jpg")
    parser.add_argument('--patch_size', default=512, type=int)
    parser.add_argument('--downsample', action='store_true')
    parser.add_argument('--truncation', action='store_true')
    parser.add_argument('--slide_format', default='svs')
    args = parser.parse_args()
    
    # Load all JPEG images and sort by their coordinates
    images = []
    for filename in glob.glob(f'./{args.project}/AdaSlide_{args.lambda_cond}_decoded/enhanced/*.{args.patch_format}'):
        x, y = extract_coords(filename, args.downsample)
        image = pyvips.Image.new_from_file(filename, access='sequential')
        
        if args.patch_format == "png" and image.bands == 3:
            image = image.bandjoin(255)
        
        images.append((x, y, image))

    # Sort images by coordinates to ensure correct order
    images.sort(key=lambda img: (img[1], img[0]))  # Sort by y, then x

    if args.truncation:
        # Calculate the total size of the stitched image
        max_x = max(img[0] + img[2].width for img in images)
        max_y = max(img[1] + img[2].height for img in images)

        min_x = min(img[0] + img[2].width for img in images)
        min_y = min(img[1] + img[2].height for img in images)
        
        final_image = pyvips.Image.black(max_x - min_x + args.patch_size, max_y - min_y + args.patch_size)
        wsi_size = [max_x - min_x + args.patch_size, max_y - min_y + args.patch_size]
    else:
        slide_file = glob.glob(f"{args.project}/WSI/*.{args.slide_format}")[0]
        slide_ = openslide.OpenSlide(slide_file)

        min_x = 0
        min_y = 0
        
        if args.downsample:
            max_x = int(slide_.level_dimensions[0][0] / 2)
            max_y = int(slide_.level_dimensions[0][1] / 2)
        else:
            max_x = int(slide_.level_dimensions[0][0])
            max_y = int(slide_.level_dimensions[0][1])
        
        final_image = pyvips.Image.black(max_x, max_y)
        wsi_size = [max_x, max_y]
        

    print(f"Size of Image: {wsi_size[0]}-{wsi_size[1]}")
    
    # Composite images onto the base image at their respective coordinates
    for x, y, img in tqdm(images):
        if args.truncation:
            final_image = final_image.insert(img, int(x - min_x), int(y - min_y))
        else:
            final_image = final_image.insert(img, x, y)
        
    # Save the stitched image as a pyramidal TIFF
    result_path = f'./{args.project}/AdaSlide_{args.lambda_cond}_decoded/reconstrctued'
    os.makedirs(result_path, exist_ok=True)
    result_fname = f'{result_path}/AdaSlide_{args.lambda_cond}_Enhanced.tif'
    final_image.tiffsave(result_fname,
                         pyramid=True, tile=True, tile_width=1024, tile_height=1024, 
                         compression='jpeg', bigtiff=True, Q=90)

    slide = openslide.OpenSlide(result_fname)
    thumbnail = slide.get_thumbnail((1024, 1024)).save(f"{result_path}/AdaSlide_{args.lambda_cond}_Enhanced.png")
    
    print("Pyramidal TIFF image stitched and saved successfully.")