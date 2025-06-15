import os
import parmap
import shutil
import argparse
import pandas as pd

def generate_results_folder(project, lambda_cond):
    os.makedirs(f"{project}/AdaSlide_{lambda_cond}/HR", exist_ok=True)
    os.makedirs(f"{project}/AdaSlide_{lambda_cond}/LR-x4", exist_ok=True)

def read_CDA_inference(infer_path, lambda_cond):
    df = pd.read_csv(infer_path).loc[:, ["fname", lambda_cond]]
    df.columns = ["fname", "action"]
    
    return df

def copy_files_to_destination(idx, infer_df, lambda_cond, patch_format="jpg"):
    fname = infer_df.loc[idx, "fname"]
    action = infer_df.loc[idx, "action"]
 
    if patch_format == "hybrid":
        lr_patch_format = ".png"
        hr_patch_format = ".jpg"
    else:
        lr_patch_format = f".{patch_format}"
        hr_patch_format = f".{patch_format}"
 
    if action == 1:
        source = fname.replace("/HR/", "/LR-x4/") + lr_patch_format
        dest = fname.replace("/HR/", f"/AdaSlide_{lambda_cond}/LR-x4/") + lr_patch_format
        shutil.copy(source, dest)
    elif action == 0:
        source = fname + hr_patch_format
        dest = fname.replace("/HR/", f"/AdaSlide_{lambda_cond}/HR/") + hr_patch_format
        shutil.copy(source, dest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project')
    parser.add_argument('--inference_file', help="Specify inference file (csv file format with binary label)")
    parser.add_argument('--lambda_cond', default="lambda_000")
    parser.add_argument('--patch_format', default="jpg", choices=["jpg", "png", "hybrid"])
    args = parser.parse_args()

    generate_results_folder(args.project, args.lambda_cond)
    infer_df = read_CDA_inference(
        args.inference_file,
        args.lambda_cond
    )
    
    parmap.map(
        copy_files_to_destination, 
        range(len(infer_df)),
        infer_df,
        args.lambda_cond, 
        args.patch_format,
        pm_pbar=True, 
        pm_processes=32
    )