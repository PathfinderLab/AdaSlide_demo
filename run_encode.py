import os
import glob
import h5py
import parmap
import argparse
import openslide
from PIL import Image
from tqdm.auto import tqdm

def generate_results_folder(project):
    dest_hr = f"{project}/HR"
    dest_lr = f"{project}/LR-x4"
    os.makedirs(dest_hr, exist_ok=True)
    os.makedirs(dest_lr, exist_ok=True)
    
def generate_patches(coord, project, slide_format, patch_format, level_of_interest=0, is_donwsample=False):
    slide_file = glob.glob(f"{project}/WSI/*.{slide_format}")[0]
    slide = openslide.OpenSlide(slide_file)

    if is_donwsample == True:
        patch = slide.read_region(tuple(coord), level_of_interest, (1024, 1024))
        patch = patch.resize((512, 512))
    else:
        patch = slide.read_region(tuple(coord), level_of_interest, (512, 512))
    
    if patch_format == "jpg":
        patch = patch.convert("RGB")
        
    patch.save(f'{project}/HR/{project}_{coord[0]}-{coord[1]}.{patch_format}')
    
    lr_patch = patch.resize((128, 128))
    lr_patch.save(f'{project}/LR-x4/{project}_{coord[0]}-{coord[1]}.{patch_format}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project')
    parser.add_argument('--slide_format', default="svs")
    parser.add_argument('--patch_format', default="jpg")
    parser.add_argument('--processes', type=int, default=16)
    parser.add_argument('--level_of_interest', type=int, default=0)
    parser.add_argument('--is_downsample', type=bool, default=False)
    args = parser.parse_args()
    
    generate_results_folder(args.project)
    h5_fname = glob.glob(f"{args.project}/CLAM_prepared/patches/*.h5")[0]
    
    with h5py.File(h5_fname, 'r') as f:
        coords = f['coords']
        
        parmap.map(
            generate_patches, 
            coords, 
            args.project,
            args.slide_format, 
            args.patch_format,
            args.level_of_interest, 
            args.is_downsample, 
            pm_processes=args.processes, 
            pm_pbar=True
        ) 