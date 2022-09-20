import logging
from pathlib import Path
import numpy as np
import torch
import json
from copy import deepcopy
from tensorboardX import SummaryWriter
from tqdm import tqdm

from lib.test import test
from lib.train import train, validate
from lib.utils import get_torch_device, np_encoder,\
     save_stacked_array, load_stacked_arrays
from lib.solvers import initialize_optimizer, initialize_scheduler
from lib.pc_utils import read_plyfile
import MinkowskiEngine as ME

HEURISTICS = ["random", "mc", "gt"]

def load_idx(train_data_loaders, idx_data):
    for i, train_data_loader in enumerate(train_data_loaders):
        heur_idx = idx_data[i]
        for j in range(len(heur_idx)):
            train_data_loader.dataset.selected_masks[j] = heur_idx[j]
    
    return train_data_loaders

def dump_idx(train_data_loader, config, heur):
    mask_export = [mask for mask in train_data_loader.dataset.selected_masks]
    save_stacked_array(config.log_dir + "/{}.npz".format(heur), mask_export, axis=0)

def choose_new_points(model, train_data_loader, config, heur, device):
    logging.info("Choosing new training points with {} heuristic...".format(heur))
    for index in tqdm(range(len(train_data_loader.dataset.selected_masks))):
        mask_idx = np.arange(train_data_loader.dataset.selected_masks[index].shape[0])
        mask_idx = mask_idx[~train_data_loader.dataset.selected_masks[index]]
        if len(mask_idx) != 0:
            if len(mask_idx) > config.npoints:
                if heur == "random":
                    selected_points = np.random.choice(mask_idx, size=config.npoints, replace=False)
                else:
                    file_name = train_data_loader.dataset.data_root / train_data_loader.dataset.data_paths[index]
                    pcd = read_plyfile(file_name)
                    coords, feats, labels= pcd[:, :3], pcd[:, 3:-1], pcd[:, -1]
                    coords, feats, labels = coords[mask_idx], feats[mask_idx], labels[mask_idx]
                    quantized_coords = np.floor(coords / train_data_loader.dataset.VOXEL_SIZE)
                    inds, inverse = ME.utils.sparse_quantize(quantized_coords, return_index=True, return_inverse=True)
                    quantized_coords = quantized_coords[inds]
                    feats = feats[inds]

                    coord_feat = [(quantized_coords, feats)]
                    with torch.no_grad():
                        coordinates_, features_  = list(zip(*coord_feat))
                        coordinates, features = ME.utils.sparse_collate(coordinates_, features_)

                        # Normalize features and create a sparse tensor
                        features = (features - 0.5).float()

                        sinput = ME.SparseTensor(features, coords=coordinates).to(device)
                    if heur == "mc":
                        sum_probs = 0
                        for _ in range(config.mc_iters):
                            with torch.no_grad():
                                soutput = model(sinput, dropout_prob=0.25)
                                logits = soutput.F
                                probs = torch.nn.functional.softmax(logits, dim=1)
                                sum_probs += probs
                        avg_prob = sum_probs / config.mc_iters
                        entropy = (-avg_prob * avg_prob.log()).sum(1).cpu().numpy()
                        entropy_orig = entropy[inverse]
                        selected_points = np.argsort(entropy_orig)[-config.npoints:]
                    else:
                        with torch.no_grad():
                            soutput = model(sinput)
                        _, pred = soutput.F.max(1)
                        pred = pred.cpu().numpy()
                        pred_orig = pred[inverse]
                        wrong_preds = ~np.equal(pred_orig, labels)
                        if wrong_preds.sum() < config.npoints:
                            selected_points = mask_idx[wrong_preds]
                        else:
                            selected_points = np.random.choice(mask_idx[wrong_preds], size=config.npoints, replace=False)
            else:
                selected_points = mask_idx
            train_data_loader.dataset.selected_masks[index][selected_points] = True
    logging.info("Finished choosing points for {} heuristic!".format(heur))
    return train_data_loader
    

def active_learning(NetClass, num_in_channel, num_labels, train_data_loaders, val_data_loader, config):
    device = get_torch_device(config.is_cuda)
    base_logdir = Path(config.log_dir)
    start_cycle = 0
    sub_cycle_ind = 0
    resume = False
    if base_logdir.exists():
        save_idx = []
        for heur in HEURISTICS:
            data_idx = load_stacked_arrays(base_logdir / "{}.npz".format(heur), axis=0)
            save_idx.append(data_idx)
        train_data_loaders = load_idx(train_data_loaders, save_idx)
        heur_exps = []
        for heur in HEURISTICS:
            heur_last_exp = max([int(run.name.split('_')[-1]) for run in (base_logdir / heur).glob('run_*')])
            heur_exps.append(heur_last_exp)
        sub_cycle_ind = np.argmin(heur_exps)
        start_cycle = heur_exps[sub_cycle_ind]
        resume=True
    writer = SummaryWriter(log_dir=config.log_dir)
    for cycle in range(start_cycle, config.num_cycles):
        cycle_perf = {}
        for heur_index in range(0, len(HEURISTICS)):
            if cycle == start_cycle and heur_index < sub_cycle_ind:
                continue
            heur_config = deepcopy(config)
            heur_config.log_dir = str(base_logdir / HEURISTICS[heur_index] / "run_{}".format(cycle))
            model = NetClass(num_in_channel, num_labels, config)
            model.to(device)
            if resume:
                heur_config.resume = str(base_logdir / HEURISTICS[heur_index] / "run_{}".format(cycle))
            train(model, train_data_loaders[heur_index], val_data_loader, heur_config)
            dump_idx(train_data_loaders[heur_index], config, HEURISTICS[heur_index])
            model = NetClass(num_in_channel, num_labels, config)
            state = torch.load(heur_config.log_dir + '/weights.pth')
            model.load_state_dict(state['state_dict'])
            model.to(device)
            model.eval()
            train_data_loaders[heur_index] = choose_new_points(model, train_data_loaders[heur_index], config, HEURISTICS[heur_index], device)
            _, _, _, v_mIoU = test(model, val_data_loader, heur_config)
            cycle_perf[HEURISTICS[heur_index]] = v_mIoU
        writer.add_scalars('active_learning/mIoU', cycle_perf, cycle + 1)

