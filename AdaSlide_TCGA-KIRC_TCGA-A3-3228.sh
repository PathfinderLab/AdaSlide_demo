#!/bin/bash

echo "\nStep 1: Preprocessing ..."
python ./CLAM/create_patches_fp.py --source /data/sample_dataset/TCGA-KIRC/TCGA-A3-3228/WSI --step_size 1024 --patch_size 1024 --patch --seg --stitch --save_dir /data/sample_dataset/TCGA-KIRC/TCGA-A3-3228/CLAM_prepared --patch_level 0

echo "\nStep 2: Encoding patches ..."
python ./run_encode.py --project /data/sample_dataset/TCGA-KIRC/TCGA-A3-3228 --slide_format svs --patch_format hybrid --processes 32 --level_of_interest 0 --is_downsample True

echo "\nStep 3: Running inference with CDA ..."
python ./infer_CDA.py --project /data/sample_dataset/TCGA-KIRC/TCGA-A3-3228 --patch_format hybrid

echo "\nStep 4: Building compressed features ..."
python ./build_compressed_features.py --project /data/sample_dataset/TCGA-KIRC/TCGA-A3-3228 --inference_file ./CompressionDecisionAgent/inferences/CompAgent_inference_task-TCGA-A3-3228.csv --lambda_cond lambda_010 --patch_format hybrid
python ./build_compressed_features.py --project /data/sample_dataset/TCGA-KIRC/TCGA-A3-3228 --inference_file ./CompressionDecisionAgent/inferences/CompAgent_inference_task-TCGA-A3-3228.csv --lambda_cond lambda_025 --patch_format hybrid
python ./build_compressed_features.py --project /data/sample_dataset/TCGA-KIRC/TCGA-A3-3228 --inference_file ./CompressionDecisionAgent/inferences/CompAgent_inference_task-TCGA-A3-3228.csv --lambda_cond lambda_050 --patch_format hybrid
python ./build_compressed_features.py --project /data/sample_dataset/TCGA-KIRC/TCGA-A3-3228 --inference_file ./CompressionDecisionAgent/inferences/CompAgent_inference_task-TCGA-A3-3228.csv --lambda_cond lambda_075 --patch_format hybrid
python ./build_compressed_features.py --project /data/sample_dataset/TCGA-KIRC/TCGA-A3-3228 --inference_file ./CompressionDecisionAgent/inferences/CompAgent_inference_task-TCGA-A3-3228.csv --lambda_cond lambda_100 --patch_format hybrid

echo "\nStep 5: Decoding features ..."
python ./run_decode.py --project /data/sample_dataset/TCGA-KIRC/TCGA-A3-3228 --lambda_cond lambda_010 --patch_format hybrid --FIE_weight FIE/net_g_latest.pth

echo "\nStep 6: Generating WSI-readable image ..."
ulimit -n 100000
python ./gen_WSI_readerable_image.py --project /data/sample_dataset/TCGA-KIRC/TCGA-A3-3228 --lambda_cond lambda_010 --patch_format hybrid --slide_format svs --truncation --downsample

echo "\nAll steps completed successfully!"