#!/bin/bash

echo "\nStep 1: Preprocessing ..."
python ./CLAM/create_patches_fp.py --source sample_dataset/Camelyon16/test_069/WSI --step_size 512 --patch_size 512 --patch --seg --stitch --save_dir sample_dataset/Camelyon16/test_069/CLAM_prepared --patch_level 0

echo "\nStep 2: Encoding patches ..."
python ./run_encode.py --project sample_dataset/Camelyon16/test_069 --slide_format tif --patch_format jpg --processes 32 --level_of_interest 0 --is_downsample False

echo "\nStep 3: Running inference with CDA ..."
python ./infer_CDA.py --project sample_dataset/Camelyon16/test_069 --patch_format jpg

echo "\nStep 4: Building compressed features ..."
python ./build_compressed_features.py --project sample_dataset/Camelyon16/test_069 --inference_file ./CompressionDecisionAgent/inferences/CompAgent_inference_task-test_069.csv --lambda_cond lambda_050

echo "\nStep 5: Decoding features ..."
python ./run_decode.py --project sample_dataset/Camelyon16/test_069 --lambda_cond lambda_050 --patch_format jpg --FIE_weight FIE/net_g_latest.pth

echo "\nStep 6: Generating WSI-readable image ..."
ulimit -n 100000
python ./gen_WSI_readerable_image.py --project sample_dataset/Camelyon16/test_069 --lambda_cond lambda_050 --patch_format jpg --slide_format tif --truncation 