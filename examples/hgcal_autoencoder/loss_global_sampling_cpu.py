###############################################################################
# Imports
###############################################################################
import os
import time
import copy
import yaml
import torch
import random
import datetime
import argparse
import json
import numpy as np
from tqdm import tqdm
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt

from autoencoder import AutoencoderNeqModel, EncoderNeqModel
from telescope_pt import telescopeMSE8x8

from pyhessian import hessian
from loss_landscapes_pinn import *
from loss_landscapes_pinn.metrics import *

from argparse import ArgumentParser

from dataset import HGCalAutoencoderDataset
from autoencoder import AutoencoderNeqModel
from ensemble_models import (
    VotingAutoencoderNeqModel,
    SnapshotAutoencoderNeqModel,
    FGEAutoencoderNeqModel,
    AdaBoostAutoencoderNeqModel,
    BaggingAutoencoderNeqModel,
)
from training_methods import (
    train,
    test,
    train_snapshot_ensemble,
    train_fge,
    train_adaboost,
    train_bagging,
)
from telescope_pt import move_constants_to_gpu
from logicnets.nn import SparseLinearNeq

###############################################################################
# Configurations
###############################################################################

ENSEMBLING_METHODS = ["voting", "snapshot", "fge", "adaboost", "bagging"]
FLAG = False
DEVICE = "cpu"

###############################################################################
# Functions
###############################################################################
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        # print("m_orig: ", m_orig.data.shape)
        # print("m_perb: ", m_perb.data.shape)
        # print("d: ", d.shape)
        # print("alpha: ", alpha)
        # try:
        if m_orig.data.shape == d.shape and m_perb.data.shape == d.shape:
            m_perb.data = m_orig.data + alpha * d
        # except:
        #     pass
    return model_perb

def set_params(model_perb, weights_perb):
    for m_perb, w_perb in zip(model_perb.parameters(), weights_perb):
        m_perb.data = w_perb
    return model_perb

# adapted from https://discuss.pytorch.org/t/how-to-flatten-and-then-unflatten-all-model-parameters/34730/2
def flatten_params(parameters, return_params=False):
    """
    flattens all parameters into a single column vector. Returns the dictionary to recover them
    :param: parameters: a generator or list of all the parameters
    :return: a dictionary: {"params": [#params, 1],
    "indices": [(start index, end index) for each param] **Note end index in uninclusive**

    """
    l = [torch.flatten(p) for p in parameters]
    indices = []
    s = 0
    for p in l:
        size = p.shape[0]
        indices.append((s, s+size))
        s += size
    flat = torch.cat(l).view(-1, 1)
    # print("flat: ", flat)
    # print("indices: ", indices)
    if return_params:
        return {"flat": flat, "indices": indices, "params": l}
    return {"flat": flat, "indices": indices}

def recover_flattened(flat_params, indices, model):
    """
    Gives a list of recovered parameters from their flattened form
    :param flat_params: [#params, 1]
    :param indices: a list detaling the start and end index of each param [(start, end) for param]
    :param model: the model that gives the params with correct shapes
    :return: the params, reshaped to the ones in the model, with the same order as those in the model
    """
    l = [torch.Tensor(flat_params[s:e]) for (s, e) in indices]
    # print("model: ", model)
    # print("l: ", l)
    # print(model.parameters())
    for i, p in enumerate(model.parameters()):
        # print(f"p: {p.shape}")
        # print(p)
        if len(p.size()) == 0:
            continue
        l[i] = l[i].view(*p.shape)
    return l

# define the function to get the model coordinates
def get_xy(point, origin,  vec_x, vec_y):
    point = point.detach().cpu().numpy()
    # vec_x = vec_x.detach().cpu().numpy()
    # vec_y = vec_y.detach().cpu().numpy()
    return np.array([np.dot(point - origin, vec_x), np.dot(point - origin, vec_y)])

def get_xy_numpy(point, origin,  vec_x, vec_y):
    # vec_x = vec_x.detach().cpu().numpy()
    # vec_y = vec_y.detach().cpu().numpy()
    return np.array([np.dot(point - origin, vec_x), np.dot(point - origin, vec_y)])

def get_one_direction_distance(point, origin, vec):
    # point = point.detach().cpu().numpy()
    # vec = vec.detach().cpu().numpy()
    return np.dot(point - origin, vec)

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

def main(args):
    # Load hyperparameters config file
    with open(args.hparams_config, "r") as f:
        config = yaml.safe_load(f)
    # Set random seeds
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if args.gpu:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    print(f"Global seed set to {seed}!")

    # Create experiment directory
    experiment_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Fetch datasets
    dataset = {}
    dataset["train"] = HGCalAutoencoderDataset(
        args.data_file, 
        data_dir=args.data_dir, 
        process_data=args.process_data, # Only need to pass once
        split="train",
    ) 
    dataset["valid"] = HGCalAutoencoderDataset(
        args.data_file, 
        data_dir=args.data_dir, 
        split="test",
    ) 
    dataset["test"] = HGCalAutoencoderDataset(
        args.data_file, 
        data_dir=args.data_dir, 
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
                config, num_models=ensemble_size, single_model_mode=args.train,
            )
        elif ensemble_method == "fge":
            model = FGEAutoencoderNeqModel(
                config, num_models=ensemble_size, single_model_mode=args.train,
            )
            cycle = config["ensemble_hp"]["cycle"]
            lr1   = config["ensemble_hp"]["lr1"]
            lr2   = config["ensemble_hp"]["lr2"]
        elif ensemble_method == "adaboost":
            model = AdaBoostAutoencoderNeqModel(
                config, 
                len(dataset["train"]), 
                num_models=ensemble_size,
                single_model_mode=args.train
            )
            independent = config["ensemble_hp"]["independent"]
            if "fixed_decoder" in config["ensemble_hp"].keys():
                fixed_decoder = config["ensemble_hp"]["fixed_decoder"]
            if "ensemble_hp" in config.keys():
                fixed_sparsity_mask = config["ensemble_hp"]["fixed_sparsity_mask"]
        elif ensemble_method == "bagging":
            model = BaggingAutoencoderNeqModel(
                config, num_models=ensemble_size, single_model_mode=args.train,
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
            model.encoder, experiment_dir, args.experiment_name
        )
        print(f"Encoder LUTCost = {encoder_lut_cost}")

     # Push model and constants to GPU if necessary
    # if args.gpu:
    #     model.cuda()
    #     move_constants_to_gpu()

    # Train
    if args.train:
        if args.checkpoint:
            print(f"Loading pre-trained checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=torch.device(DEVICE)) 
            model.load_state_dict(checkpoint["model_dict"])
        train_params = {
            "gpu": args.gpu,
            "wd": config["wd"],
            "lr": config["lr"],
            "epochs": config["epochs"],
            "num_workers": args.num_workers,
            "experiment_dir": experiment_dir,
            "batch_size": config["batch_size"],
            "finetune_epochs": finetune_epochs,
            "warm_restart_freq": config["warm_restart_freq"],
            # Sequential ensemble learning hyperparemeter
            "independent": independent,
            "fixed_decoder": fixed_decoder,
            "fixed_sparsity_mask": fixed_sparsity_mask, 
            # FGE hyperparameters
            "cycle": cycle,
            "lr1": lr1,
            "lr2": lr2,
        }
        # Log experiment hyperparameters
        hparams_log = os.path.join(
            experiment_dir,"hparams.yml"
        )
        with open(hparams_log, "w") as f:
            yaml.dump(config, f)
        # Start training
        if ensemble_method == "voting":
            train(model, dataset, train_params)
        elif ensemble_method == "snapshot":
            train_snapshot_ensemble(model, config, dataset, train_params)
        elif ensemble_method == "fge":
            train_fge(model, config, dataset, train_params)
        elif ensemble_method == "adaboost":
            train_adaboost(model, config, dataset, train_params)
        elif ensemble_method == "bagging":
            train_bagging(model, config, dataset, train_params)
        else:
            train(model, dataset, train_params)

    # Evaluate model
    evaluate_model = False
    if args.evaluate:
        if args.checkpoint:
            # Evaluate given checkpoint
            evaluate_model = True
            print(f"Evaluating model saved at: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=torch.device(DEVICE)) 
            model.load_state_dict(checkpoint["model_dict"])
            if ensemble_method == "adaboost":
                model.model_weights = checkpoint["model_weights"]
                if args.gpu:
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
    elif args.train:
        evaluate_model = True 
        if ensemble_method in ENSEMBLING_METHODS and "voting" not in ensemble_method:
            ensemble_ckpt_path = os.path.join(
                experiment_dir, 'last_ensemble_ckpt.pth'
            )
            print(f"Evaluating last ensemble saved at: {ensemble_ckpt_path}")
            best_checkpoint = torch.load(ensemble_ckpt_path, map_location=torch.device(DEVICE)) 
            if ensemble_method == "adaboost":
                model.model_weights = best_checkpoint["model_weights"]
                if args.gpu:
                    model.model_weights.cuda()
            model.single_model_mode = False
        else:
            ckpt_path = os.path.join(experiment_dir, 'best_loss.pth')
            print(f"Evaluating best model saved at: {ckpt_path}")
            best_checkpoint = torch.load(ckpt_path, map_location=torch.device(DEVICE)) 
        model.load_state_dict(best_checkpoint["model_dict"])

    if evaluate_model:
        print("Evaluating model")
        # Need val_sum to compute EMD
        _, val_sum = dataset["test"].get_val_max_and_sum()
        test_loader = torch.utils.data.DataLoader(
            dataset["test"], 
            num_workers=args.num_workers, 
            batch_size=config["batch_size"], 
            pin_memory=True,
            shuffle=False, 
        )
        # test_loss, avg_emd = test(
        #     model, test_loader, val_sum, args.gpu, compute_emd=True,
        # )
        # test_loss, avg_emd = test(
        #     model, test_loader, val_sum, False, compute_emd=True,
        # )
        # eval_tag = "_eval" if args.evaluate else ""
        # os.makedirs(experiment_dir, exist_ok=True)
        # test_results_log = os.path.join(
        #     experiment_dir, 
        #     args.experiment_name \
        #     + f"_loss={test_loss:.3f}" + eval_tag + "_emd.txt"
        # )
        # with open(test_results_log, "w") as f:
        #     f.write(str(avg_emd))
        #     f.close()

        ###############################################################################
        # Calculate the Hessian loss
        ###############################################################################

        criterion = telescopeMSE8x8
        # criterion = torch.nn.CrossEntropyLoss()

        train_loader = torch.utils.data.DataLoader(
            dataset["train"], 
            num_workers=args.num_workers,
            batch_size=config["batch_size"], 
            pin_memory=True,
            shuffle=False, 
        )

        for x in train_loader:
            break
        inputs, targets = x, x

        # # make the model a copy of the original model
        model.eval()
        model_experiment = copy.deepcopy(model)
        model_experiment.eval()

        # extract individual ensemble member checkpoints
        models = []
        individual_checkpoints = model.encoder_ensemble
        individual_checkpoint = EncoderNeqModel(config, input_length=64, output_length=16)
        print("Number of individual checkpoints: ", len(individual_checkpoints))
        for k in range(len(individual_checkpoints)):
            # print("Individual checkpoint: ", individual_checkpoint)
            # Load the checkpoint
            individual_checkpoint = individual_checkpoints[k]
            individual_checkpoint.load_state_dict(individual_checkpoint.state_dict())
            individual_checkpoint.eval()
            models.append(individual_checkpoint)

            # model_experiment.encoder_ensemble = torch.nn.ModuleList()
            # encoder = copy.deepcopy(individual_checkpoint)
            # model_experiment.encoder_ensemble.append(encoder)
            # model_experiment.eval()

            # models.append(model_experiment)

        print("Number of individual models: ", len(models))

        ###############################################################################
        # Compute global directions using models in the plane
        ###############################################################################

        # collect weights and make sure we are appending weights correctly
        weight_dicts = []
        weights = []
        for i, model in enumerate(models):
            # {"params": flat, "indices": indices}
            weight_dict = flatten_params(model.parameters(), return_params=True)
            w_flat = weight_dict['flat'].detach().cpu().numpy().ravel()
            
            weight_dicts.append(weight_dict)
            weights.append(w_flat)

        print(f"Done collecting {len(weights)} model weights")
        for i, w_i in enumerate(weights):
            print(f"{i} // {np.shape(w_i)}")
        for i, w_i in enumerate(weight_dicts):
            print(f"{i} // {len(w_i['params'])}")

        # define (N-1) global directions

        global_directions = []
        global_norms = []

        # define (N-1) global directions 
        for i, w_i in enumerate(weights[1:], start=1):

            # subtract w_0 from w_i
            v_i = w_i - weights[0]
            v_i_orig = copy.deepcopy(v_i)
            print(f"v_i_orig: {np.shape(v_i_orig)}")
            print("v_i_orig: ", v_i_orig[:30])

            # make orthogonal to other directions
            for v_j in global_directions:
                v_i = v_i - np.dot(v_j, v_i) * v_j
            print(f"v_i: {np.shape(v_i)}")
            print("v_i: ", v_i[:30])

            # # compare v_i before and after making it orthogonal
            print(f"{np.linalg.norm(v_i_orig - v_i)=}")
            print(f"{np.dot(v_i_orig, v_i)=}")

            # normalize
            v_i_norm = np.linalg.norm(v_i)
            # v_i_norm = 1.0
            v_i /= v_i_norm

            # store directions and norms
            global_directions.append(v_i)
            global_norms.append(v_i_norm)
            
        # TODO: how to unflatten directions so we can perturb model??
        # TODO: can we reshape global_directions based on weight dicts
        global_directions_flat = copy.deepcopy(global_directions)
        global_directions = []
        for i, v_i in enumerate(global_directions_flat):
            print(f"{i} // {len(v_i)}")
            print(f"{i} // {np.shape(v_i)}")
            print(weight_dicts[i]['indices'])
            v_i = recover_flattened(v_i, weight_dicts[i]['indices'], models[i])
            global_directions.append(v_i)

        print(f"Done unflattening {len(global_directions)} global directions")
        for i, v_i in enumerate(global_directions):
            print(f"{i} // {len(v_i)}")
        
        ###############################################################################
        # Fix the dimensions
        ###############################################################################
        DIM = args.dim
        if len(global_directions) < 2:
            exit("Error: Number of global directions is less than 2. Exiting...")
        if DIM > len(global_directions):
            print(f"Warning: DIM={DIM} is larger than the number of global directions. Setting DIM to {len(global_directions)}")
            DIM = len(global_directions)

        ###############################################################################
        # Compute coordinates of the loaded models
        ###############################################################################
        model_coords = []

        for i, w in enumerate(weights):
            current_coords = []
            for j, v in enumerate(global_directions_flat):
                distance = get_one_direction_distance(w, weights[0], v)
                current_coords.append(distance)
            model_coords.append(current_coords)

        print(f"Model coordinates: {model_coords}")
        print("Size of model coordinates: ", len(model_coords))

        ###############################################################################
        # Get the whole distance box
        ###############################################################################
        distance_box = []
        for i in range(len(current_coords)):
            left_right_distance = max([_[i] for _ in model_coords]) - min([_[i] for _ in model_coords])
            left_border = min([_[i] for _ in model_coords]) - args.box_size * left_right_distance
            right_border = max([_[i] for _ in model_coords]) + args.box_size * left_right_distance
            print(f"Direction {i}: {left_border} to {right_border}")
            distance_box.append((left_border, right_border))
        print(f"Distance box: {distance_box}")

        ###############################################################################
        # Get initial model
        ###############################################################################

        model_init = None
        model_init = copy.deepcopy(models[0])
        model_init.eval()

        # ### TODO: take average of all models
        # def get_weighted_model(model, weight=1.0):
        #     for m in model.parameters():
        #         m.data = torch.mul(m.data, weight)
        #     return model    

        # if args.model_init == "centroid":
        #     print(f"Using the centroid as the initial model.")

        #     # compute average over all models (i.e., find the centroid)
        #     model_init = copy.deepcopy(models[0])
        #     model_init = get_weighted_model(model_init, weight=(1 / len(models)))
        #     for model in models[1:]:
        #         model_other = copy.deepcopy(model)
        #         model_other = get_weighted_model(model_other, weight=(1 / len(models)))
        #         # update model_init
        #         for m_i, m_o in zip(model_init.parameters(), model_other.parameters()):
        #             m_i.data = m_i.data + m_o.data

        #     # do we need to re-evaluate?
        #     model_init.eval()

        # elif int(args.model_init) in args.seeds:
        #     print(f"Using seed={args.model_init} as the initial model.")

        #     ### force override and just use first model
        #     # print("Seeds are: ", args.seeds)
        #     # model_init_index = args.seeds.index(args.model_init)
        #     model_init_index = int(args.model_init)
        #     model_init = copy.deepcopy(models[model_init_index])
        #     model_init.eval()

        # else: 
        #     print(f"[!] model_init={args.model_init} not recognized...")
        #     print(f"[!] Using seed=0 as the initial model.")

        #     ### force override and just use first model
        #     model_init = copy.deepcopy(models[0])
        #     model_init.eval()

        # make the model a copy
        model_perb = copy.deepcopy(model_init)
        model_perb.eval()

        ###############################################################################
        # Random sampling the loss landscape coordinates
        ###############################################################################

        set_seed(0) # for weight initialization

        # Generate the loss values array using BFS
        POINTS = args.points
        # Create a coordinate sampling array for loss values
        loss_coordinates_list = []
        pbar = tqdm(total=POINTS, desc="Generating sampling coordinates in the subspace")
        for j in range(len(model_coords)):
            loss_coordinates_list.append(model_coords[j])
            pbar.update(1)
        while len(loss_coordinates_list) < POINTS:
            t = ()
            for i in range(len(distance_box)):
                t += (np.random.uniform(distance_box[i][0], distance_box[i][1]),)
            if t not in loss_coordinates_list:
                loss_coordinates_list.append(t)
                pbar.update(1)
        pbar.close()
        loss_coordinates = np.asarray(loss_coordinates_list)
        print("Generated coordinates: ", loss_coordinates.shape)

        # calculate the distance between the generated coordinates and original model
        distance_matrix_list = []
        for i in range(len(loss_coordinates)):
            distance = np.linalg.norm(loss_coordinates[i] - model_coords[0])
            distance_matrix_list.append("{:.2f}".format(distance))
        distance_matrix = np.asarray(distance_matrix_list)
        print("Distance matrix: ", distance_matrix.shape)
        
        ###############################################################################
        # Evaluate the loss along points on the grid
        ###############################################################################

        # start timer
        # start = time.time()

        # send the model and datato GPU if available
        if FLAG == True:
            # model.cuda()
            model_init.cuda()
            model_perb.cuda()

        # Create a data matrix to store loss values
        data_matrix = np.empty([POINTS, 1], dtype=float)

        # Fill array with initial value (e.g., -1)
        data_matrix.fill(-1)

        # calculate the loss values
        for j in tqdm(range(POINTS), desc="Calculating sampling loss values in the subspace"):
            # adjust the model and fill with a loss with corresponding model parameters
            next_pos = tuple(loss_coordinates[j])
            # print(f"Next position: {next_pos}")
            model_current = copy.deepcopy(model_init)
            for i in range(DIM):
                # print(DIM)
                global_direction = global_directions[i]
                model_perb = get_params(model_current, model_perb, global_direction, next_pos[i])
                model_current = copy.deepcopy(model_perb)
            
            # compute the loss for the current model
            model_experiment.encoder_ensemble = torch.nn.ModuleList()
            encoder = copy.deepcopy(model_current)
            model_experiment.encoder_ensemble.append(encoder)
            model_experiment.eval()
            inputs_hat = model_experiment(inputs)
            loss = criterion(inputs_hat, targets)
        
            # if j < len(model_coords):
                # print(f"Loss value at {j}th model: {loss}")
            data_matrix[j] = loss.detach().numpy()

        ###############################################################################
        # Fix the loss values
        ###############################################################################

        loss_values = copy.deepcopy(data_matrix)
        print("Max value of loss values: ", np.max(loss_values))
        print("Min value of loss values: ", np.min(loss_values))
        if args.vmax_pct != -1:
            vmax = np.percentile(loss_values.ravel(), args.vmax_pct)
            loss_values_orig = loss_values.copy()
            loss_values[loss_values > vmax] = vmax
            print(f"vmax = {vmax} (vmax_pct={args.vmax_pct})")

        ###############################################################################
        # Save the results
        ###############################################################################

        # plot the loss values if visualize is set
        if DIM == 2 and args.visualize:
            x = loss_coordinates[:, 0]
            y = loss_coordinates[:, 1]
            plt.scatter(x, y, c=loss_values, cmap='viridis')
            plt.colorbar()
            for i in range(len(model_coords)):
                x_ = model_coords[i][0]
                y_ = model_coords[i][1]
                loss_ = loss_values[i]
                plt.scatter(x_, y_, c='red')

            plt.title(f'Global Loss landscape')
            plt.savefig('loss_landscapes_global/' + str(args.experiment_name) + '_hessian_loss_landscape_' + f'boxsize{args.box_size}_max_pct{args.vmax_pct}_dim{DIM}_points{POINTS}.png')
            if args.show_plots:
                plt.show()
            plt.clf()

        if DIM == 3 and args.visualize:
            ###############################################################################
            # Calculate the full loss cubes with fixed step sizes
            ###############################################################################
            STEPS = args.steps

            lams_x = np.linspace(distance_box[0][0], distance_box[0][1], STEPS)
            lams_y = np.linspace(distance_box[1][0], distance_box[1][1], STEPS)
            lams_z = np.linspace(distance_box[2][0], distance_box[2][1], STEPS)

            full_cube_coordinates = []
            for i in range(STEPS):
                for j in range(STEPS):
                    for k in range(STEPS):
                        t = (lams_x[i], lams_y[j], lams_z[k])
                        full_cube_coordinates.append(t)

            full_cube_coordinates = np.asarray(full_cube_coordinates)
            print("Generated full cube coordinates: ", full_cube_coordinates.shape)

            # calculate the loss values
            full_cube_matrix = np.empty([STEPS**DIM, 1], dtype=float)
            full_cube_matrix.fill(-1)

            # make the model a copy
            model_init = copy.deepcopy(models[0])
            model_init.eval()
            # make the model another copy
            model_perb = copy.deepcopy(model_init)
            model_perb.eval()

            # calculate the loss values
            for j in tqdm(range(STEPS**DIM), desc="Calculating sampling loss values in the full cube"):
                # adjust the model and fill with a loss with corresponding model parameters
                next_pos = tuple(full_cube_coordinates[j])
                model_current = copy.deepcopy(model_init)
                for i in range(DIM):
                    global_direction = global_directions[i]
                    model_perb = get_params(model_current, model_perb, global_direction, next_pos[i])
                    model_current = copy.deepcopy(model_perb)

                # compute the loss for the current model
                model_experiment.encoder_ensemble = torch.nn.ModuleList()
                encoder = copy.deepcopy(model_current)
                model_experiment.encoder_ensemble.append(encoder)
                model_experiment.eval()
                inputs_hat = model_experiment(inputs)
                loss = criterion(inputs_hat, targets)
                # print(f"Loss value at coordinate {j}: {loss}")
                full_cube_matrix[j] = loss.detach().numpy()

            ###############################################################################
            # Save scatter plot of the full cube
            ###############################################################################

            x = full_cube_coordinates[:, 0]
            y = full_cube_coordinates[:, 1]
            z = full_cube_coordinates[:, 2]

            print(f"Generated full cube coordinates: {x.shape}, {y.shape}, {z.shape}")
            print(f"Generated full cube loss values: {full_cube_matrix.shape}")

            fig = plt.figure(figsize = (10, 7))
            ax = plt.axes(projection ="3d")
            ax.scatter3D(x, y, z, c=full_cube_matrix, cmap= "viridis")

            plt.title(f'Global Loss landscape')
            plt.savefig('loss_landscapes_global_3D/' + str(args.experiment_name) + '_loss_landscape_' + f'boxsize{args.box_size}_max_pct{args.vmax_pct}_dim{DIM}_steps{STEPS}.png')
            if args.show_plots:
                plt.show()
            plt.clf()

            ###############################################################################
            # Save the full cube results
            ###############################################################################

            # save the loss values
            np.save('loss_landscapes_global_3D/' + str(args.experiment_name) + '_loss_landscape_' + f'boxsize{args.box_size}_max_pct{args.vmax_pct}_dim{DIM}_steps{STEPS}.npy', full_cube_matrix.ravel())
            # save the coordinates
            np.save('loss_landscapes_global_3D/' + str(args.experiment_name) + '_loss_landscape_' + f'boxsize{args.box_size}_max_pct{args.vmax_pct}_dim{DIM}_steps{STEPS}_coords.npy', full_cube_coordinates)
            # save the loss with its corresponding model coordinates into a json
            with open('loss_landscapes_global_3D/' + str(args.experiment_name) + '_loss_landscape_' + f'boxsize{args.box_size}_max_pct{args.vmax_pct}_dim{DIM}_steps{STEPS}.json', "w") as f:
                json.dump({"loss_values": full_cube_matrix.tolist(), "loss_coordinates": full_cube_coordinates.tolist()}, f)
        
        ###############################################################################
        # Save the sampling cube results
        ###############################################################################
        # save the loss values
        np.save('loss_landscapes_global/' + str(args.experiment_name) + '_hessian_loss_landscape_' + f'boxsize{args.box_size}_max_pct{args.vmax_pct}_dim{DIM}_points{POINTS}.npy', data_matrix.ravel())
        # save the coordinates
        np.save('loss_landscapes_global/' + str(args.experiment_name) + '_hessian_loss_landscape_' + f'boxsize{args.box_size}_max_pct{args.vmax_pct}_dim{DIM}_points{POINTS}_coords.npy', loss_coordinates)
        # save the loss with its corresponding model coordinates into a json
        with open('loss_landscapes_global/' + str(args.experiment_name) + '_hessian_loss_landscape_' + f'boxsize{args.box_size}_max_pct{args.vmax_pct}_dim{DIM}_points{POINTS}.json', "w") as f:
            json.dump({"loss_values": data_matrix.tolist(), "loss_coordinates": loss_coordinates.tolist(), "numbers of models": len(models)}, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    # Dataset args
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--data_file", type=str, default="data.npy")
    parser.add_argument("--process_data", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="./autoencoder_test")
    parser.add_argument("--experiment_name", type=str, default="autoencoder")
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument(
        "--hparams_config", 
        type=str, 
        default=None, 
        help="yaml containing hyperparameters for building and training a model"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None, 
        help="model checkpoint"
    )
    parser.add_argument('--visualize', type=bool, default=True, help='Visualize the solution.')
    parser.add_argument('--show-plots', default=False, help='Visualize the solution.')
    parser.add_argument('--box-size', default=1.0, type=float, help='Size of the box to visualize.')
    parser.add_argument('--dim', default=3, help='dimension for hessian loss values calculation')
    parser.add_argument('--steps', default=50, help='steps for hessian loss values calculation')
    parser.add_argument('--points', default=5000, help='total points while sampling for hessian loss values calculation')
    parser.add_argument('--vmax-pct', type=float, default=30, help='Clamp values above this percentile when visualizing')
    args = parser.parse_args()
    main(args)
