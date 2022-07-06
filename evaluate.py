# Copyright (C) 2021-2022 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import sys
import argparse
from pathlib import Path
import yaml 
import ast

import torch
from torchvision import transforms

from how.utils import io_helpers, logging, download
from how.stages.evaluate import eval_asmk
from examples.demo_how import _overwrite_cirtorch_path, DATASET_URL

import fire_network

def evaluate_demo(demo_eval, evaluation, globals):
    globals["device"] = torch.device("cpu")
    if demo_eval['gpu_id'] is not None:
        globals["device"] = torch.device(("cuda:%s" % demo_eval['gpu_id']))

    # Handle net_path when directory
    net_path = Path(demo_eval['exp_folder']) / demo_eval['net_path']
    if net_path.is_dir() and (net_path / "epochs/model_best.pth").exists():
        net_path = net_path / "epochs/model_best.pth"

    # Load net
    state = torch.load(net_path, map_location='cpu')
    state['net_params']['pretrained'] = None # no need for imagenet pretrained model
    net = fire_network.init_network(**state['net_params']).to(globals['device'])
    net.load_state_dict(state['state_dict'])
    globals["transform"] = transforms.Compose([transforms.ToTensor(), \
                transforms.Normalize(**dict(zip(["mean", "std"], net.runtime['mean_std'])))])

    # Eval
    eval_asmk(net, evaluation['inference'], globals, **evaluation['local_descriptor'])


def main(args):
    """Argument parsing and parameter preparation for the demo"""
    # Arguments
    parser = argparse.ArgumentParser(description="FIRe evaluation.")
    parser.add_argument('parameters', type=str, help="Relative path to a yaml file that contains parameters.")
    parser.add_argument("--experiment", "-e", metavar="NAME", dest="experiment")
    parser.add_argument("--model-load", "-ml", metavar="PATH", dest="demo_eval.net_path")
    parser.add_argument("--data-folder", metavar="PATH", dest="demo_eval.data_folder")
    parser.add_argument("--exp-folder", metavar="PATH", dest="demo_eval.exp_folder")
    parser.add_argument("--features-num", metavar="NUM",
                        dest="evaluation.inference.features_num", type=int)
    parser.add_argument("--scales", metavar="SCALES", dest="evaluation.inference.scales",
                        type=ast.literal_eval)
    args = parser.parse_args(args)

    # Load yaml params
    package_root = Path(__file__).resolve().parent
    parameters_path = args.parameters
    params = io_helpers.load_params(parameters_path)
    # Overlay with command-line arguments
    for arg, val in vars(args).items():
        if arg not in {"command", "parameters"} and val is not None:
            io_helpers.dict_deep_set(params, arg.split("."), val)
            
    # Resolve experiment name
    exp_name = params.pop("experiment")
    if not exp_name:
        exp_name = Path(parameters_path).name[:-len(".yml")]

    # Resolve data folders
    globals = {}
    globals["root_path"] = (package_root / params['demo_eval']['data_folder'])
    globals["root_path"].mkdir(parents=True, exist_ok=True)
    _overwrite_cirtorch_path(str(globals['root_path']))
    globals["exp_path"] = (package_root / params['demo_eval']['exp_folder']) / exp_name
    globals["exp_path"].mkdir(parents=True, exist_ok=True)
    # Setup logging
    globals["logger"] = logging.init_logger(globals["exp_path"] / f"eval.log")

    # Run demo
    io_helpers.save_params(globals["exp_path"] / f"eval_params.yml", params)
    params['evaluation']['global_descriptor'] = dict(datasets=[])
    download.download_for_eval(params['evaluation'], params['demo_eval'], DATASET_URL, globals)
    
    evaluate_demo(**params, globals=globals)


if __name__ == "__main__":
    main(sys.argv[1:])