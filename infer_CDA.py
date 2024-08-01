import os
import argparse
from tqdm.auto import tqdm
from collections import defaultdict

from CompressionDecisionAgent.models.model import load_compression_agent
from CompressionDecisionAgent.models.model_interface import CompressionAgentInference
from CompressionDecisionAgent.utils.data import get_inference_image_flist, define_inference_dataset_and_dataloader
from CompressionDecisionAgent.utils.logging import genereate_prediction_dataframe

def get_models():
    comp_agent_list = [
        "lambda_010",
        "lambda_025",
        "lambda_050",
        "lambda_075",
        "lambda_100"
    ]
    weights_path = {
        "lambda_010": "CompressionAgent_lambda-0.10.pt",
        "lambda_025": "CompressionAgent_lambda-0.25.pt",
        "lambda_050": "CompressionAgent_lambda-0.50.pt",
        "lambda_075": "CompressionAgent_lambda-0.75.pt",
        "lambda_100": "CompressionAgent_lambda-1.00.pt"    
    }

    comp_agent_weight_path = "CompressionDecisionAgent/weights/"

    models = {}
    for comp_agent in comp_agent_list:
        model = load_compression_agent(os.path.join(comp_agent_weight_path, weights_path[comp_agent]))
        model_interface = CompressionAgentInference(model)
        models[comp_agent] = model_interface
    
    return models, comp_agent_list

def run_prediction(project, inference_dataloader, comp_agent_list, result_path="CompressionDecisionAgent/inferences/", get_prob=False):
    results_dict = defaultdict(list)

    for batch in tqdm(inference_dataloader, total=len(inference_dataloader)):
        fname, image = batch
        results_dict["fname"].extend(fname)
        
        for comp_agent in comp_agent_list:
            if get_prob == False:
                action, _ = models[comp_agent].predict(image)
                results_dict[comp_agent].extend(action)
            elif get_prob == True:
                _, prob = models[comp_agent].predict(image)
                results_dict[comp_agent].extend(prob)
            
    results_dict["lambda_inf"] = [0] * len(results_dict["fname"])
    results_dict["lambda_000"] = [1] * len(results_dict["fname"])

    # Save results
    os.makedirs(result_path, exist_ok=True)
    result = genereate_prediction_dataframe(results_dict)
    if get_prob == False:
        result.to_csv(f"{result_path}/CompAgent_inference_task-{project}.csv", index=False)
    elif get_prob == True:
        result.to_csv(f"{result_path}/CompAgent_inference_task-{project}_prob.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project')
    parser.add_argument('--patch_format', default="jpg")
    parser.add_argument('--get_prob', default=False, type=bool)
    args = parser.parse_args()

    flist = get_inference_image_flist(f"{args.project}/HR/*.{args.patch_format}")
    inference_dataloader = define_inference_dataset_and_dataloader(flist)
    
    models, comp_agent_list = get_models()
    run_prediction(args.project, inference_dataloader, comp_agent_list, get_prob=args.get_prob)