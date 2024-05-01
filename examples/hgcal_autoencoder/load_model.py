import os
import yaml
import torch
import random
import numpy as np

from dataset import HGCalAutoencoderDataset
from autoencoder import AutoencoderNeqModel
from ensemble_models import (
    VotingAutoencoderNeqModel,
    SnapshotAutoencoderNeqModel,
    FGEAutoencoderNeqModel,
    AdaBoostAutoencoderNeqModel,
    BaggingAutoencoderNeqModel,
)
from logicnets.nn import SparseLinearNeq

ENSEMBLING_METHODS = ["voting", "snapshot", "fge", "adaboost", "bagging"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_lut_cost(model, experiment_dir, experiment_name):
    """
    Compute LUTCost of the given model
    """
    total_lut_cost = 0
    lut_cost_by_layer = {}
    for name, module in model.named_modules():
        if type(module) == SparseLinearNeq:
            lut_cost = module.lut_cost()
            print(f"{name} lut cost = {lut_cost}")
            lut_cost_by_layer[name] = lut_cost
            total_lut_cost = total_lut_cost + lut_cost
    # Log lut cost
    os.makedirs(experiment_dir, exist_ok=True)
    test_results_log = os.path.join(
        experiment_dir, 
        experiment_name + f"_lutcost.txt"
    )
    with open(test_results_log, "w") as f:
        for k in lut_cost_by_layer:
            f.write(f"{k} lut cost: {lut_cost_by_layer[k]}\n")
        f.write(f"Total LUT cost: {total_lut_cost}\n")
        f.close()
    return total_lut_cost

def load_checkpoint(checkpoint_path: str, hparams_config: str, data_path: str, data_dir: str, process_data=False, train_flag=False, evaluate_flag=True, save_dir="./autoencoder_test", experiment_name="autoencoder"):
    with open(hparams_config, "r") as f:
        config = yaml.safe_load(f)
    # Set random seeds
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    print(f"Global seed set to {seed}!")

    # Create experiment directory
    experiment_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Fetch datasets
    dataset = {}
    dataset["train"] = HGCalAutoencoderDataset(
        data_path, 
        data_dir=data_dir, 
        process_data=process_data, # Only need to pass once
        split="train",
    ) 
    dataset["valid"] = HGCalAutoencoderDataset(
        data_path, 
        data_dir=data_dir, 
        split="test",
    ) 
    dataset["test"] = HGCalAutoencoderDataset(
        data_path, 
        data_dir=data_dir, 
        split="test",
    )
    # Extra ensemble hyperparameters
    if "finetune_epochs" in config.keys():
        finetune_epochs = config["finetune_epochs"]
    else:
        finetune_epochs = 0 # By default
    # FGE default hyperparameters
    cycle = 4
    lr1 = 1e-3
    lr2 = 1e-5
    independent = True
    fixed_decoder = None
    ensemble_method = None
    fixed_sparsity_mask = False
    if "ensemble_method" in config.keys():
        # Ensemble learning
        ensemble_method = config["ensemble_method"]
        ensemble_size = config["ensemble_size"]
        if ensemble_method == "voting":
            if "ensemble_hp" in config.keys():
                fixed_sparsity_mask = config["ensemble_hp"]["fixed_sparsity_mask"]
            model = VotingAutoencoderNeqModel(
                config, 
                num_models=ensemble_size, 
                fixed_sparsity_mask=fixed_sparsity_mask
            )

        elif ensemble_method == "snapshot":
            model = SnapshotAutoencoderNeqModel(
                config, num_models=ensemble_size, single_model_mode=train_flag,
            )
        elif ensemble_method == "fge":
            model = FGEAutoencoderNeqModel(
                config, num_models=ensemble_size, single_model_mode=train_flag,
            )
            cycle = config["ensemble_hp"]["cycle"]
            lr1   = config["ensemble_hp"]["lr1"]
            lr2   = config["ensemble_hp"]["lr2"]
        elif ensemble_method == "adaboost":
            model = AdaBoostAutoencoderNeqModel(
                config, 
                len(dataset["train"]), 
                num_models=ensemble_size,
                single_model_mode=train_flag
            )
            independent = config["ensemble_hp"]["independent"]
            if "fixed_decoder" in config["ensemble_hp"].keys():
                fixed_decoder = config["ensemble_hp"]["fixed_decoder"]
            if "ensemble_hp" in config.keys():
                fixed_sparsity_mask = config["ensemble_hp"]["fixed_sparsity_mask"]
        elif ensemble_method == "bagging":
            model = BaggingAutoencoderNeqModel(
                config, num_models=ensemble_size, single_model_mode=train_flag,
            )
            independent = config["ensemble_hp"]["independent"]
            if "fixed_decoder" in config["ensemble_hp"].keys():
                fixed_decoder = config["ensemble_hp"]["fixed_decoder"]
        else:
            raise ValueError(f"Unsupported ensemble method: {ensemble_method}")
    else: # Single model learning
        # Build model
        model = AutoencoderNeqModel(config)
        # Compute LUTCost of whole model
        encoder_lut_cost = get_lut_cost(
            model.encoder, experiment_dir, experiment_name
        )
        print(f"Encoder LUTCost = {encoder_lut_cost}")

    # Evaluate model
    evaluate_model = False
    if evaluate_flag:
        if checkpoint_path:
            # Evaluate given checkpoint
            evaluate_model = True
            print(f"Evaluating model saved at: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(DEVICE)) 
            model.load_state_dict(checkpoint["model_dict"])
            if ensemble_method == "adaboost":
                model.model_weights = checkpoint["model_weights"]
                if DEVICE == "cuda":
                    model.model_weights.cuda()
            if (
                ensemble_method == "snapshot"
                or ensemble_method == "fge"
                or ensemble_method == "adaboost"
                or ensemble_method == "bagging"
            ):
                model.single_model_mode = False
        else:
            raise ValueError(
                "No checkpoint provided for evaluation. " \
                "Provide a path to checkpoint argument, " \
                "i.e., --checkpoint CHECKPOINT_PATH"
            ) 
    elif train_flag:
        evaluate_model = True 
        if ensemble_method in ENSEMBLING_METHODS and "voting" not in ensemble_method:
            ensemble_ckpt_path = os.path.join(
                experiment_dir, 'last_ensemble_ckpt.pth'
            )
            print(f"Evaluating last ensemble saved at: {ensemble_ckpt_path}")
            best_checkpoint = torch.load(ensemble_ckpt_path, map_location=torch.device(DEVICE)) 
            if ensemble_method == "adaboost":
                model.model_weights = best_checkpoint["model_weights"]
                if DEVICE == "cuda":
                    model.model_weights.cuda()
            model.single_model_mode = False
        else:
            ckpt_path = os.path.join(experiment_dir, 'best_loss.pth')
            print(f"Evaluating best model saved at: {ckpt_path}")
            best_checkpoint = torch.load(ckpt_path, map_location=torch.device(DEVICE)) 
        model.load_state_dict(best_checkpoint["model_dict"])
    
    print(f"Model loaded successfully!")
    return model
