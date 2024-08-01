import os
import argparse
from tqdm.auto import tqdm
from collections import defaultdict

from models.model import load_compression_agent
from models.model_interface import CompressionAgentInference
from utils.config import load_config
from utils.data import get_inference_image_flist, define_inference_dataset_and_dataloader
from utils.logging import genereate_prediction_dataframe, print_inference_result_stat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="specify config file")
    args = parser.parse_args()
    
    # Load config file
    config = load_config(args.config)
    
    task_param = config["task"]
    
    task_name = task_param["name"]
    task_path = task_param["path"]
    task_data_format = task_param["format"]
    num_classes = task_param["num_classes"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    
    # Define Dataset and Loader
    flist = get_inference_image_flist(task_path + task_data_format)
    inference_dataloader = define_inference_dataset_and_dataloader(flist, 
                                                                   batch_size=batch_size, 
                                                                   num_workers=num_workers)
    
    # Define Compression Agents
    comp_agent_config = config["comp_agent"]    
    comp_agent_weight_path = comp_agent_config["weight_path"]
    comp_agent_list = comp_agent_config["agent_list"]
    
    models = {}
    
    for comp_agent in comp_agent_list:
        model = load_compression_agent(os.path.join(comp_agent_weight_path, comp_agent_config[comp_agent]))
        model_interface = CompressionAgentInference(model)
        models[comp_agent] = model_interface
    
    # Do inference
    results_dict = defaultdict(list)
    
    for batch in tqdm(inference_dataloader, total=len(inference_dataloader)):
        fname, image = batch
        results_dict["fname"].extend(fname)
        
        for comp_agent in comp_agent_list:
            action = models[comp_agent].predict(image)
            results_dict[comp_agent].extend(action)
    
    # Save results
    result_path = config["result"]["path"]
    os.makedirs(result_path, exist_ok=True)
    result = genereate_prediction_dataframe(results_dict)
    result.to_csv(f"{result_path}/CompAgent_inference_task-{task_name}.csv", index=False)
    
    print_inference_result_stat(result)