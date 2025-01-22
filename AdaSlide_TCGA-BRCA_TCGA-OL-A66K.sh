#!/bin/bash

echo "\nStep 1: Preprocessing ..."
python ./CLAM/create_patches_fp.py --source sample_dataset/TCGA-BRCA/TCGA-OL-A66K/WSI --step_size 1024 --patch_size 1024 --patch --seg --stitch --save_dir sample_dataset/TCGA-BRCA/TCGA-OL-A66K/CLAM_prepared --patch_level 0

echo "\nStep 2: Encoding patches ..."
python ./run_encode.py --project sample_dataset/TCGA-BRCA/TCGA-OL-A66K --slide_format svs --patch_format jpg --processes 32 --level_of_interest 0 --is_downsample True

echo "\nStep 3: Running inference with CDA ..."
python ./infer_CDA.py --project sample_dataset/TCGA-BRCA/TCGA-OL-A66K --patch_format jpg

echo "\nStep 4: Building compressed features ..."
python ./build_compressed_features.py --project sample_dataset/TCGA-BRCA/TCGA-OL-A66K --inference_file ./CompressionDecisionAgent/inferences/CompAgent_inference_task-TCGA-OL-A66K.csv --lambda_cond lambda_050

echo "\nStep 5: Decoding features ..."
python ./run_decode.py --project sample_dataset/TCGA-BRCA/TCGA-OL-A66K --lambda_cond lambda_050 --patch_format jpg --FIE_weight FIE/net_g_latest.pth

echo "\nStep 6: Generating WSI-readable image ..."
ulimit -n 50000
python ./gen_WSI_readerable_image.py --project sample_dataset/TCGA-BRCA/TCGA-OL-A66K --lambda_cond lambda_050 --patch_format jpg --slide_format svs --truncation --downsample

echo "\nAll steps completed successfully!"