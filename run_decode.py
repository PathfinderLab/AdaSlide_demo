import os
import cv2
import glob
import shutil
import parmap
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from basicsr.archs.rrdbnet_arch import RRDBNet 

def generate_results_folder(project, lambda_cond):
    os.makedirs(f"./{project}/AdaSlide_{lambda_cond}_decoded/enhanced", exist_ok=True)

def load_FID_ESRGAN(ckpt="./FIE/net_g_latest.pth", device="cuda:0"):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    model.load_state_dict(torch.load(ckpt)['params'], strict=True)
    model.eval()
    model.to("cuda:0")

    return model

def copy_files(hr_path):
    source = hr_path
    dest = hr_path.replace(f"/AdaSlide_{args.lambda_cond}/HR/", f"/AdaSlide_{args.lambda_cond}_decoded/enhanced/")     
    shutil.copy(source, dest)

def read_image(lr_path, device="cuda:0"):
    lr_image = cv2.imread(lr_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    lr_image = torch.from_numpy(np.transpose(lr_image[:, :, [2, 1, 0]], (2, 0, 1))).float()
    lr_image = lr_image.unsqueeze(0).to(device)
    
    return lr_image

def post_processing(output):
    output = output.data.squeeze().float().detach().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    output = cv2.GaussianBlur(output, (3, 3), 0)
    
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project')
    parser.add_argument('--lambda_cond', default="lambda_000")
    parser.add_argument('--patch_format', default="jpg")
    parser.add_argument('--FIE_weight', default="./FIE/net_g_latest.pth")
    args = parser.parse_args()

    generate_results_folder(args.project, args.lambda_cond)
    
    hr_flist = glob.glob(f"{args.project}/AdaSlide_{args.lambda_cond}/HR/*.{args.patch_format}")
    parmap.map(copy_files, hr_flist, pm_pbar=False, pm_processes=16)
    
    model = load_FID_ESRGAN(args.FIE_weight)
    
    lr_flist = glob.glob(f"{args.project}/AdaSlide_{args.lambda_cond}/LR-x4/*.{args.patch_format}")
    
    with torch.no_grad():
        for lr_path in tqdm(lr_flist):
            dest = lr_path.replace(f"/AdaSlide_{args.lambda_cond}/LR-x4/", f"/AdaSlide_{args.lambda_cond}_decoded/enhanced/")
            
            lr_image = read_image(lr_path)
            output = model(lr_image)
            output = post_processing(output)
            
            cv2.imwrite(dest, output)