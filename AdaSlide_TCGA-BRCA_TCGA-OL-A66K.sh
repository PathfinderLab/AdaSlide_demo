#!/bin/bash

echo "\nStep 1: Preprocessing ..."
python ./CLAM/create_patches_fp.py --source ~/Workspace/sample_dataset/TCGA-BRCA/TCGA-OL-A66K/WSI --step_size 1024 --patch_size 1024 --patch --seg --stitch --save_dir ~/Workspace/sample_dataset/TCGA-BRCA/TCGA-OL-A66K/CLAM_prepared --patch_level 0

echo "\nStep 2: Encoding patches ..."
python ./run_encode.py --project ~/Workspace/sample_dataset/TCGA-BRCA/TCGA-OL-A66K --slide_format svs --patch_format hybrid --processes 32 --level_of_interest 0 --is_downsample True

echo "\nStep 3: Running inference with CDA ..."
python ./infer_CDA.py --project ~/Workspace/sample_dataset/TCGA-BRCA/TCGA-OL-A66K --patch_format hybrid

echo "\nStep 4: Building compressed features ..."
python ./build_compressed_features.py --project ~/Workspace/sample_dataset/TCGA-BRCA/TCGA-OL-A66K --inference_file ./CompressionDecisionAgent/inferences/CompAgent_inference_task-TCGA-OL-A66K.csv --lambda_cond lambda_010 --patch_format hybrid
python ./build_compressed_features.py --project ~/Workspace/sample_dataset/TCGA-BRCA/TCGA-OL-A66K --inference_file ./CompressionDecisionAgent/inferences/CompAgent_inference_task-TCGA-OL-A66K.csv --lambda_cond lambda_025 --patch_format hybrid
python ./build_compressed_features.py --project ~/Workspace/sample_dataset/TCGA-BRCA/TCGA-OL-A66K --inference_file ./CompressionDecisionAgent/inferences/CompAgent_inference_task-TCGA-OL-A66K.csv --lambda_cond lambda_050 --patch_format hybrid
python ./build_compressed_features.py --project ~/Workspace/sample_dataset/TCGA-BRCA/TCGA-OL-A66K --inference_file ./CompressionDecisionAgent/inferences/CompAgent_inference_task-TCGA-OL-A66K.csv --lambda_cond lambda_075 --patch_format hybrid
python ./build_compressed_features.py --project ~/Workspace/sample_dataset/TCGA-BRCA/TCGA-OL-A66K --inference_file ./CompressionDecisionAgent/inferences/CompAgent_inference_task-TCGA-OL-A66K.csv --lambda_cond lambda_100 --patch_format hybrid

echo "\nStep 5: Decoding features ..."
nohup python ./run_decode.py --project ~/Workspace/sample_dataset/TCGA-BRCA/TCGA-OL-A66K --lambda_cond lambda_010 --patch_format hybrid --FIE_weight FIE/net_g_latest.pth > decode_TCGA-OL-A66K-lambda_010.log &
nohup python ./run_decode.py --project ~/Workspace/sample_dataset/TCGA-BRCA/TCGA-OL-A66K --lambda_cond lambda_025 --patch_format hybrid --FIE_weight FIE/net_g_latest.pth > decode_TCGA-OL-A66K-lambda_025.log &
nohup python ./run_decode.py --project ~/Workspace/sample_dataset/TCGA-BRCA/TCGA-OL-A66K --lambda_cond lambda_050 --patch_format hybrid --FIE_weight FIE/net_g_latest.pth > decode_TCGA-OL-A66K-lambda_050.log &
nohup python ./run_decode.py --project ~/Workspace/sample_dataset/TCGA-BRCA/TCGA-OL-A66K --lambda_cond lambda_075 --patch_format hybrid --FIE_weight FIE/net_g_latest.pth > decode_TCGA-OL-A66K-lambda_075.log &
nohup python ./run_decode.py --project ~/Workspace/sample_dataset/TCGA-BRCA/TCGA-OL-A66K --lambda_cond lambda_100 --patch_format hybrid --FIE_weight FIE/net_g_latest.pth > decode_TCGA-OL-A66K-lambda_100.log &

echo "\nStep 6: Generating WSI-readable image ..."
ulimit -n 10000
python ./gen_WSI_readerable_image.py --project ~/Workspace/sample_dataset/TCGA-BRCA/TCGA-OL-A66K --lambda_cond lambda_050 --patch_format hybrid --slide_format svs --downsample

echo "\nAll steps completed successfully!"