import os
import pickle
import pandas as pd

def save_results(project_name, comp_ratio, actions, dice_score):
    result_fname = "./results/compression_results.csv"
    
    if os.path.exists(result_fname):
        df = pd.read_csv(result_fname)       
        df_new = pd.DataFrame(comp_ratio, index=[0])
        df_new.loc[:, "project_name"] = project_name
        df_new = df_new.loc[:, ["project_name", "overall", "low", "high"]]
        
        df = df.append(df_new).sort_values("project_name").reset_index(drop=True)
        df.to_csv(result_fname, index=False)
    else:
        df_new = pd.DataFrame(comp_ratio, index=[0])
        df_new.loc[:, "project_name"] = project_name
        df_new = df_new.loc[:, ["project_name", "overall", "low", "high"]]
        df_new.to_csv(result_fname, index=False)
        
    pickle_dict = {
        "Action": actions,
        "Dice": dice_score
    }
        
    with open(f"./results/pkl_files/{project_name}.pkl", "wb") as f:
        pickle.dump(pickle_dict, f)
    
    print(f"Results of {project_name} was successfully saved!")
    
def genereate_prediction_dataframe(result):
    return pd.DataFrame(result)

def print_inference_result_stat(result):
    print(result.apply(pd.Series.value_counts).iloc[:2, 1:].astype(int))