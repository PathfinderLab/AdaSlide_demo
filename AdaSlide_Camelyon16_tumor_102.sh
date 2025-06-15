#!/bin/bash
echo "\nStep 1: Preprocessing ..."
python ./CLAM/create_patches_fp.py --source ~/Workspace/sample_dataset/Camelyon16/tumor_102/WSI --step_size 1024 --patch_size 1024 --patch --seg --stitch --save_dir ~/Workspace/sample_dataset/Camelyon16/tumor_102/CLAM_prepared --patch_level 0

echo "\nStep 2: Encoding patches ..."
python ./run_encode.py --project ~/Workspace/sample_dataset/Camelyon16/tumor_102 --slide_format tif --patch_format hybrid --processes 32 --level_of_interest 0 --is_downsample True

echo "\nStep 3: Running inference with CDA ..."
python ./infer_CDA.py --project ~/Workspace/sample_dataset/Camelyon16/tumor_102 --patch_format hybrid

echo "\nStep 4: Building compressed features ..."
python ./build_compressed_features.py --project ~/Workspace/sample_dataset/Camelyon16/tumor_102 --inference_file ./CompressionDecisionAgent/inferences/CompAgent_inference_task-tumor_102.csv --lambda_cond lambda_010 --patch_format hybrid
python ./build_compressed_features.py --project ~/Workspace/sample_dataset/Camelyon16/tumor_102 --inference_file ./CompressionDecisionAgent/inferences/CompAgent_inference_task-tumor_102.csv --lambda_cond lambda_025 --patch_format hybrid
python ./build_compressed_features.py --project ~/Workspace/sample_dataset/Camelyon16/tumor_102 --inference_file ./CompressionDecisionAgent/inferences/CompAgent_inference_task-tumor_102.csv --lambda_cond lambda_050 --patch_format hybrid
python ./build_compressed_features.py --project ~/Workspace/sample_dataset/Camelyon16/tumor_102 --inference_file ./CompressionDecisionAgent/inferences/CompAgent_inference_task-tumor_102.csv --lambda_cond lambda_075 --patch_format hybrid
python ./build_compressed_features.py --project ~/Workspace/sample_dataset/Camelyon16/tumor_102 --inference_file ./CompressionDecisionAgent/inferences/CompAgent_inference_task-tumor_102.csv --lambda_cond lambda_100 --patch_format hybrid

echo "\nStep 5: Decoding features ..."
CUDA_VISIBLE_DEVICES=0 nohup python ./run_decode.py --project ~/Workspace/sample_dataset/Camelyon16/tumor_102 --lambda_cond lambda_010 --patch_format hybrid --FIE_weight FIE/net_g_latest.pth > decode_tumor_102-lambda_010.log &
CUDA_VISIBLE_DEVICES=0 nohup python ./run_decode.py --project ~/Workspace/sample_dataset/Camelyon16/tumor_102 --lambda_cond lambda_025 --patch_format hybrid --FIE_weight FIE/net_g_latest.pth > decode_tumor_102-lambda_025.log &
CUDA_VISIBLE_DEVICES=1 nohup python ./run_decode.py --project ~/Workspace/sample_dataset/Camelyon16/tumor_102 --lambda_cond lambda_050 --patch_format hybrid --FIE_weight FIE/net_g_latest.pth > decode_tumor_102-lambda_050.log &
CUDA_VISIBLE_DEVICES=1 nohup python ./run_decode.py --project ~/Workspace/sample_dataset/Camelyon16/tumor_102 --lambda_cond lambda_075 --patch_format hybrid --FIE_weight FIE/net_g_latest.pth > decode_tumor_102-lambda_075.log &
CUDA_VISIBLE_DEVICES=1 nohup python ./run_decode.py --project ~/Workspace/sample_dataset/Camelyon16/tumor_102 --lambda_cond lambda_100 --patch_format hybrid --FIE_weight FIE/net_g_latest.pth > decode_tumor_102-lambda_100.log &

echo "\nStep 6: Generating WSI-readable image ..."
ulimit -n 10000
python ./gen_WSI_readerable_image.py --project ~/Workspace/sample_dataset/Camelyon16/tumor_102 --lambda_cond lambda_100 --patch_format hybrid --slide_format tif --downsample

echo "\nAll steps completed successfully!"