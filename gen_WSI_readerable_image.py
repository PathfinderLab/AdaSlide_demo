import os
import re
import glob
import time
import pyvips
import parmap
import argparse
import warnings
import openslide
from tqdm.auto import tqdm

import threading
import itertools
import sys

pyvips.cache_set_max(0)
pyvips.cache_set_max_mem(0)

# Function to extract coordinates from filename
def extract_coords(filename, is_downsample=False):
    base = filename.split("_")[-1].split(".")[0]
    x, y = base.split("-")
    x, y = int(x), int(y)
    
    if is_downsample == True:
        x = int(x / 2)
        y = int(y / 2)
    
    return x, y

def spinner_task(stop_event):
    spinner = itertools.cycle(['|', '/', '-', '\\'])
    while not stop_event.is_set():
        sys.stdout.write(f"\rSaving stitched image... {next(spinner)}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\rSaving stitched image... Done!    \n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project')
    parser.add_argument('--lambda_cond', default="lambda_000")
    parser.add_argument('--patch_format', default="jpg", choices=["jpg", "png", "hybrid"])
    parser.add_argument('--patch_size', default=512, type=int)
    parser.add_argument('--downsample', action='store_true')
    parser.add_argument('--slide_format', default='svs')
    parser.add_argument('--level_of_interest', default=1, type=int)
    args = parser.parse_args()
    
    # Load all JPEG images and sort by their coordinates
    images = []
    
    if args.patch_format != "hybrid":
        flist = glob.glob(f'{args.project}/AdaSlide_{args.lambda_cond}_decoded/enhanced/*.{args.patch_format}')
    elif args.patch_format == "hybrid":
        flist = glob.glob(f'{args.project}/AdaSlide_{args.lambda_cond}_decoded/enhanced/*.jpg') + \
                glob.glob(f'{args.project}/AdaSlide_{args.lambda_cond}_decoded/enhanced/*.png')

    print(len(flist), "images found.")
    
    for filename in flist:
        x, y = extract_coords(filename, args.downsample)
        # image = pyvips.Image.new_from_file(filename, access='sequential')
        with open(filename, "rb") as f:
            buffer = f.read()
        image = pyvips.Image.new_from_buffer(buffer, "", access='sequential')
        
        if filename.split(".")[-1] == "png" and image.bands == 3:
            image = image.bandjoin(255)
        
        if image.bands == 4:
            image = image[:3]
        
        images.append((x, y, image))

    # Sort images by coordinates to ensure correct order
    images.sort(key=lambda img: (img[1], img[0]))  # Sort by y, then x

    slide_file = glob.glob(f"{args.project}/WSI/*.{args.slide_format}")[0]
    slide_ = openslide.OpenSlide(slide_file)

    min_x = 0
    min_y = 0
    
    # TODO : Add truncation option
    if args.downsample:
        max_x = int(slide_.level_dimensions[0][0] / 2)
        max_y = int(slide_.level_dimensions[0][1] / 2)
    
    final_image = pyvips.Image.black(max_x, max_y)
    wsi_size = [max_x, max_y]

    print(f"Size of Image: {wsi_size[0]}-{wsi_size[1]}")
    
    # Composite images onto the base image at their respective coordinates
    for x, y, img in tqdm(images):  
        final_image = final_image.insert(img, x, y)
        
    # Save the stitched image as a pyramidal TIFF
    result_path = f'{args.project}/AdaSlide_{args.lambda_cond}_decoded/reconstructed'
    os.makedirs(result_path, exist_ok=True)
    result_fname = f'{result_path}/AdaSlide_{args.lambda_cond}_Enhanced.tif'

    warnings.warn("This process may take a while depending on the number of images and their sizes.")
    
    print("\n\nStart generating a pyramidal TIFF.")
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=spinner_task, args=(stop_spinner,))
    spinner_thread.start()
    
    tic = time.time()
    final_image.tiffsave(result_fname,
                         pyramid=True, tile=True, tile_width=512, tile_height=512, 
                         compression='jpeg', bigtiff=True, Q=75)
    toc = time.time()
    stop_spinner.set()
    spinner_thread.join()
    print(f"Image saved in {toc - tic:.2f} seconds. Thank you for your patience!")
    
    print("Generating thumbnail...")
    slide = openslide.OpenSlide(result_fname)
    thumbnail = slide.get_thumbnail((1024, 1024)).save(f"{result_path}/AdaSlide_{args.lambda_cond}_Enhanced.png")
    print(f"Thumbnail generated successfully. Image saved as {result_path}/AdaSlide_{args.lambda_cond}_Enhanced.png")
    
    print("Pyramidal TIFF image stitched and saved successfully.")