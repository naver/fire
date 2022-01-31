# Copyright (C) 2021-2022 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import sys
import argparse
from pathlib import Path
import time

import torch
from torchvision import transforms

from cirtorch.datasets.datahelpers import collate_tuples
from cirtorch.datasets.traindataset import TuplesDataset

import fire_network
from losses import DecorrelationAttentionLoss, SuperfeatureLoss

from how.utils import io_helpers, logging, download, data_helpers, io_helpers, logging
from how.stages.train import Validation, set_seed, initialize_dim_reduction
from examples.demo_how import _overwrite_cirtorch_path, DATASET_URL


NUM_WORKERS = 6

# note: we only change the line that load the network!
def train(demo_train, training, validation, model, globals):
    """Demo training a network
    :param dict demo_train: Demo-related options
    :param dict training: Training options
    :param dict validation: Validation options
    :param dict model: Model options
    :param dict globals: Global options
    """
    logger = globals["logger"]
    (globals["exp_path"] / "epochs").mkdir(exist_ok=True)
    if (globals["exp_path"] / f"epochs/model_epoch{training['epochs']}.pth").exists():
        logger.info("Skipping network training, already trained")
        return

    # Global setup
    set_seed(0)
    globals["device"] = torch.device("cpu")
    if demo_train['gpu_id'] is not None:
        globals["device"] = torch.device(("cuda:%s" % demo_train['gpu_id']))

    # Initialize network
    net = fire_network.init_network(**model).to(globals["device"])
    globals["transform"] = transforms.Compose([
                transforms.RandomHorizontalFlip(p=training['transform']['flip_prob']), \
                transforms.ToTensor(), \
                transforms.Normalize(**dict(zip(["mean", "std"], net.runtime['mean_std'])))])

    with logging.LoggingStopwatch("initializing network whitening", logger.info, logger.debug):
        initialize_dim_reduction(net, globals, **training['initialize_dim_reduction'])

    # Initialize training
    optimizer, scheduler, criterion_superfeatures, criterion_attns, train_loader = \
            initialize_training(net.parameter_groups(training["optimizer"]), training, globals)
    validation = Validation(validation, globals)

    for epoch in range(training['epochs']):
        epoch1 = epoch + 1
        set_seed(epoch1)

        time0 = time.time()
        train_loss = train_epoch(train_loader, net, globals, criterion_superfeatures,
                                 criterion_attns, optimizer, epoch1)

        validation.add_train_loss(train_loss, epoch1)
        validation.validate(net, epoch1)

        scheduler.step()

        io_helpers.save_checkpoint({
            'epoch': epoch1, 'meta': net.meta, 'state_dict': net.state_dict(),
            'optimizer' : optimizer.state_dict(), 'best_score': validation.best_score[1],
            'scores': validation.scores, 'net_params': model, '_version': 'fire/2021',
        }, validation.best_score[0] == epoch1, epoch1 == training['epochs'], globals["exp_path"] / "epochs")

        logger.info(f"Epoch {epoch1} finished in {time.time() - time0:.1f}s")

def train_epoch(train_loader, net, globals, criterion_superfeatures, criterion_attns, optimizer, epoch1):
    """Train for one epoch"""
    logger = globals['logger']
    batch_time = data_helpers.AverageMeter()
    data_time = data_helpers.AverageMeter()
    losses = data_helpers.AverageMeter()
    losses_super = data_helpers.AverageMeter()
    losses_attn = data_helpers.AverageMeter()

    # Prepare epoch
    net.return_global=True
    train_loader.dataset.create_epoch_tuples(net)
    net.return_global=False
    net.train()

    end = time.time()
    for i, (inpt, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        num_images = len(inpt[0]) # number of images per tuple
        for inp, trg in zip(inpt, target):
            superfeatures = []
            attns = []
            for imi in range(num_images):
                sfs, attn, _ = net(inp[imi].to(globals["device"]))
                assert len(sfs)==1 and len(attn)==1, "Only one scale at a time during training"
                superfeatures.append( sfs[0].squeeze().T )
                attns.append( attn[0].squeeze() )

            loss_attn = criterion_attns(attns)
            loss_super = criterion_superfeatures(superfeatures, trg.to(globals["device"]))

            loss = loss_attn + loss_super
            loss.backward()

            losses_super.update(loss_super.item())
            losses_attn.update(loss_attn.item())
            losses.update(loss.item())

        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % 20 == 0 or i == 0 or (i+1) == len(train_loader):
            logger.info(f'>> Train: [{epoch1}][{i+1}/{len(train_loader)}]\t' \
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                        f'LossSuper {losses_super.val:.4f} ({losses_super.avg:.4f})\t' \
                        f'LossAttn {losses_attn.val:.4f} ({losses_attn.avg:.4f})\t')

    return losses.avg


def initialize_training(net_parameters, training, globals):
    """Initialize classes necessary for training"""
    # Need to check for keys because of defaults
    assert training['optimizer'].keys() == {"lr", "weight_decay"}
    assert training['lr_scheduler'].keys() == {"gamma"}
    assert training['loss_superfeature'].keys() == {"margin","weight"}
    assert training['loss_attention'].keys() == {"weight"}
    assert training['dataset'].keys() == {"name", "mode", "imsize", "nnum", "qsize", "poolsize"}
    assert training['loader'].keys() == {"batch_size"}

    # Adam params:  {'lr': 3e-05, 'weight_decay': 0.0001}
    optimizer = torch.optim.Adam(net_parameters, **training["optimizer"])
    # scheduler: {'gamma': 0.99}
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **training["lr_scheduler"])
    # {'margin': 1.1, 'weight': 0.02}
    criterion_superfeatures = SuperfeatureLoss(**training["loss_superfeature"]).to(globals["device"])
    criterion_attns = DecorrelationAttentionLoss(**training['loss_attention']).to(globals['device'])
    train_dataset = TuplesDataset(**training['dataset'], transform=globals["transform"])
    train_loader = torch.utils.data.DataLoader(train_dataset, **training['loader'], \
            pin_memory=True, drop_last=True, shuffle=True, collate_fn=collate_tuples, \
            num_workers=NUM_WORKERS)
    return optimizer, scheduler, criterion_superfeatures, criterion_attns, train_loader

def main(args):
    """Argument parsing and parameter preparation for the demo"""
    # Arguments
    parser = argparse.ArgumentParser(description="FIRe training.")
    parser.add_argument('parameters', type=str, help="Relative path to a yaml file that contains parameters.")
    parser.add_argument("--experiment", "-e", metavar="NAME", dest="experiment")
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
    globals["root_path"] = (package_root / params['demo_train']['data_folder'])
    globals["root_path"].mkdir(parents=True, exist_ok=True)
    _overwrite_cirtorch_path(str(globals['root_path']))
    globals["exp_path"] = (package_root / params['demo_train']['exp_folder']) / exp_name
    globals["exp_path"].mkdir(parents=True, exist_ok=True)
    # Setup logging
    globals["logger"] = logging.init_logger(globals["exp_path"] / f"train.log")

    # Run training
    io_helpers.save_params(globals["exp_path"] / f"train_params.yml", params)
    download.download_for_train(params['validation'], DATASET_URL, globals)
    if params['model']['pretrained'].startswith('http'):
        # additionally download imagenet pretrained model
        net_name = os.path.basename(params['model']['pretrained'])
        io_helpers.download_files([net_name], globals['root_path'] / "pretraining",
                                  os.path.dirname(params['model']['pretrained']) + "/",
                                  logfunc=globals["logger"].info)
        params['model']['pretrained'] = globals['root_path'] / "pretraining" / net_name
    train(**params, globals=globals)

if __name__ == "__main__":
    main(sys.argv[1:])
