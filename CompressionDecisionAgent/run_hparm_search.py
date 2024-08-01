import argparse
import pytorch_lightning as pl
from tqdm.auto import tqdm

from models.model import define_model
from models.model_interface import CompressionAgentModule, define_callbacks
from utils.config import load_config
from utils.data import define_augmentations, define_datasets_and_dataloaders
from utils.eval import evaluate
from utils.logging import save_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="specify config file")
    args = parser.parse_args()

    # Load config file
    config = load_config(args.config)
    
    # Define DataLoaders
    train_transform, valid_transform = define_augmentations()
    train_dataloader, valid_dataloader, test_dataloader = define_datasets_and_dataloaders(train_transform, valid_transform)
    
    # Define Hparam tuner
    fixed_config = config["fixed_variables"]
    tuning_config = config["tuning_variables"]
    
    num_classes = fixed_config["num_classes"]
    dice_threshold = fixed_config["dice_threshold"]
    save_top_k = fixed_config["save_top_k"]
    max_epochs = fixed_config["max_epochs"]
    gpus = fixed_config["gpus"]
    
    total_iterations = fixed_config["number_of_iterations"] * len(tuning_config["learning_rate"]) * len(tuning_config["reward_lambda"])  
    
    with tqdm(total=total_iterations, desc="Grid Search .... ") as pbar:
        for index_iter in range(fixed_config["number_of_iterations"]):
            for index_lr in range(len(tuning_config["learning_rate"])):
                for index_lambda in range(len(tuning_config["reward_lambda"])):
                    learning_rate = float(tuning_config["learning_rate"][index_lr])
                    reward_lambda_coef = tuning_config["reward_lambda"][index_lambda]

                    project_name = f"CompressAgent_lr-{learning_rate}_lambda-{reward_lambda_coef}_iter-{index_iter}"
                    
                    # Define model and interface
                    model = define_model(model_name="resnet18", num_classes=num_classes)
                    model_interface = CompressionAgentModule(
                        model=model, 
                        len_train_dataloader=len(train_dataloader),
                        len_valid_dataloader=len(valid_dataloader), 
                        learning_rate=learning_rate,
                        reward_lambda_coef=reward_lambda_coef,
                        dice_threshold=dice_threshold
                    )
                    checkpoints_callback = define_callbacks(
                        project_name=project_name, save_top_k=save_top_k
                    )
                    
                    # Define trainer and do training
                    trainer = pl.Trainer(
                        max_epochs=max_epochs, 
                        gpus=gpus, 
                        callbacks=checkpoints_callback, 
                        enable_progress_bar=True
                    )
                    trainer.fit(
                        model_interface, 
                        train_dataloader,
                        valid_dataloader
                    )
                    
                    # Do evaluation and save results
                    best_ckpt_path = trainer.checkpoint_callback.best_model_path
                    model_interface.load_from_checkpoint(
                        best_ckpt_path,
                        model=model, 
                        len_train_dataloader=1, 
                        len_valid_dataloader=1
                    )
                    
                    model_interface.model.eval()
                    comp_ratio, actions, dice_score = evaluate(
                        test_dataloader=test_dataloader, 
                        model=model, 
                        threshold=dice_threshold
                    )
                    
                    save_results(project_name, comp_ratio, actions, dice_score)
                    pbar.update(1)