# Copyright (C) 2021-2022 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import pathlib
import time
import argparse
import logging
from typing import Optional
import math
from tqdm import tqdm
from pathlib import Path
import torch
from torchvision import transforms

from cirtorch.datasets.genericdataset import ImagesFromList
from asmk.asmk_method import ASMKMethod
import how.networks.how_net
from how.utils import data_helpers, io_helpers, download
from how.stages.evaluate import _convert_checkpoint
from examples.demo_how import _overwrite_cirtorch_path, DATASET_URL
import fire_network

import kapture
from kapture.io.csv import kapture_from_dir
from kapture.io.records import get_image_fullpath
import kapture.utils.logging

logger = logging.getLogger('HOW')
os.umask(0x0002)


def _load_kapture_imagelist(kapture_root_path):
    assert os.path.isdir(kapture_root_path), "Unknown dataset "+kapture_root_path
    kdata = kapture_from_dir(kapture_root_path, None,
                             skip_list=[kapture.Keypoints,
                                        kapture.Descriptors,
                                        kapture.GlobalFeatures,
                                        kapture.Matches,
                                        kapture.Points3d,
                                        kapture.Observations],
                             tar_handlers=None)
    root = get_image_fullpath(kapture_root_path, image_filename=None)
    assert kdata.records_camera is not None
    imgs = kdata.records_camera.data_list()
    return root, imgs


def _load_model(modeltype, net_path, device):
    state = torch.load(net_path, map_location='cpu')
    if modeltype == 'how':
        state = _convert_checkpoint(state)
        net = how.networks.how_net.init_network(**state['net_params']).to(device)
    else:
        state['net_params']['pretrained'] = None  # no need for imagenet pretrained model
        net = fire_network.init_network(**state['net_params']).to(device)

    net.load_state_dict(state['state_dict'])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        **dict(zip(["mean", "std"], net.runtime['mean_std'])))])
    net = net.eval()
    return net, transform


def compute_pairs(map_path: str,
                  query_path: Optional[str],
                  modeltype: str,
                  parameters: str,
                  checkpoint: Optional[str],
                  data_folder_overwrite: Optional[str],
                  topk: int,
                  block_size: int,
                  output_path: str,
                  codebook_cache_path: Optional[str] = None,
                  ivf_cache_path: Optional[str] = None):
    """
    Compute pairs of images using FIRe or HOW

    :param map_path: input path to kapture map root directory
    :param query_path: input path to kapture query root directory (optional)
    :param modeltype: a string equal to fire or how
    :param parameters: path to a yaml file that contains evaluation parameters
        example values: (eval_fire.yml or ../how/examples/params/eccv20/eval_how_r50-_1000.yml)
    :param checkpoint: checkpoint path (overwrites demo_eval.net_path, optional)
    :param data_folder_overwrite: overwrites yml demo_eval.data_folder with this path (optional)
    :param topk: max number of images per query in output pairs
    :param block_size: max number of local features to store at once for ivf building and querying
    :param output_path: output path to pairs text file
    :param codebook_cache_path: path where the codebook will be stored (optional)
    :param ivf_cache_path: path where the ivf database will be stored (optional)
    """
    # load kapture database
    t0 = time.time()
    logger.info(f'Loading dataset {map_path}')
    dbroot, dbimgs = _load_kapture_imagelist(map_path)
    logger.debug("\t{:d} database images for dataset {:s}".format(len(dbimgs), map_path))
    logger.debug('\tdone in {:g} seconds'.format(time.time()-t0))

    t0 = time.time()
    logger.info('Preparing codebook dataset')
    package_root = Path(__file__).resolve().parent
    codebook_yaml_path = os.path.abspath(parameters)
    params = io_helpers.load_params(codebook_yaml_path)
    evaluation = params['evaluation']
    evaluation['global_descriptor'] = dict(datasets=[])
    evaluation['local_descriptor']['datasets'] = ['val_eccv20']
    inference = evaluation['inference']
    logger.debug(params)

    globals = {}
    demo_eval = params['demo_eval']
    if data_folder_overwrite is None:
        globals["root_path"] = (package_root / demo_eval['data_folder'])
    else:
        globals["root_path"] = Path(data_folder_overwrite).resolve()
    globals["root_path"].mkdir(parents=True, exist_ok=True)
    _overwrite_cirtorch_path(str(globals['root_path']))
    globals["logger"] = logger
    if checkpoint is not None:
        demo_eval['net_path'] = os.path.abspath(checkpoint)
        assert os.path.isfile(demo_eval['net_path'])
    download.download_for_eval(evaluation, demo_eval, DATASET_URL, globals)
    logger.debug('\tdone in {:g} seconds'.format(time.time()-t0))

    # load model
    t0 = time.time()
    device = torch.device("cpu")
    if demo_eval['gpu_id'] is not None:
        device = torch.device(("cuda:%s" % demo_eval['gpu_id']))

    logger.info('Loading model ' + str(demo_eval['net_path']))
    net, transform = _load_model(modeltype, demo_eval['net_path'], device)
    logger.debug('\tdone in {:g} seconds'.format(time.time()-t0))

    t0 = time.time()
    logger.info('Training codebook')
    if codebook_cache_path and os.path.exists(codebook_cache_path):
        des_train = None
    else:
        codebook_training = evaluation['local_descriptor']['codebook_training']
        images = data_helpers.load_dataset('train', data_root=globals["root_path"])[0]
        images = images[:codebook_training['images']]
        infer_opts_codebook = {"scales": codebook_training['scales'], "features_num": inference['features_num']}
        dset = ImagesFromList(root='', images=images, imsize=inference['image_size'], bbxs=None, transform=transform)
        des_train = how.networks.how_net.extract_vectors_local(net, dset, device, **infer_opts_codebook)[0]
    asmk = ASMKMethod.initialize_untrained(evaluation['local_descriptor']['asmk']).train_codebook(
        des_train, cache_path=codebook_cache_path)
    logger.debug('\tdone in {:g} seconds'.format(time.time()-t0))

    # build ivf database
    t0 = time.time()
    logger.info('Indexing database images')
    data_opts = {"imsize": inference['image_size'], "transform": transform}
    infer_opts = {"scales": inference['scales'], "features_num": inference['features_num']}

    from_cache = ivf_cache_path is not None and os.path.isfile(ivf_cache_path)
    builder = asmk.create_ivf_builder(cache_path=ivf_cache_path)
    if not from_cache:
        number_of_iteration = math.ceil(len(dbimgs) / block_size)
        for i in tqdm(range(number_of_iteration), disable=logging.getLogger().level >= logging.CRITICAL):
            dbimgs_it = dbimgs[i * block_size:(i+1)*block_size]
            dset = ImagesFromList(root=dbroot, images=dbimgs_it, bbxs=None, **data_opts)
            des0, des1, _, _, _ = how.networks.how_net.extract_vectors_local(net, dset, device, **infer_opts)
            des1 += (i * block_size)
            builder.add(des0, des1)
    asmk_dataset = asmk.add_ivf_builder(builder)
    logger.debug('\tdone in {:g} seconds'.format(time.time()-t0))

    if query_path is not None:
        t0 = time.time()
        logger.info(f'loading {query_path}')
        qroot, qimgs = _load_kapture_imagelist(query_path)
        logger.debug("\t{:d} query images for dataset {:s}".format(len(qimgs), query_path))
        logger.debug('\tdone in {:g} seconds'.format(time.time()-t0))
    else:
        qroot, qimgs = dbroot, dbimgs
    number_of_iteration_q = math.ceil(len(qimgs) / block_size)

    p = pathlib.Path(output_path)
    os.makedirs(str(p.parent.resolve()), exist_ok=True)
    with open(output_path, 'w') as fid:
        fid.write('# query_image, map_image, score\n')

    t0 = time.time()
    logger.info('computing image pairs')
    for i in tqdm(range(number_of_iteration_q), disable=logger.level >= logging.CRITICAL):
        qimgs_it = qimgs[i * block_size:(i+1)*block_size]
        qset = ImagesFromList(root=qroot, images=qimgs_it, bbxs=None, **data_opts)
        qdes0, qdes1, _, _, _ = how.networks.how_net.extract_vectors_local(net, qset, device, **infer_opts)
        _, _, ranks, _scores = asmk_dataset.query_ivf(qdes0, qdes1)

        with open(output_path, 'a') as fid:
            for iq, qimg in enumerate(qimgs_it):
                r = 0
                k = 0
                while k < topk and r < ranks.shape[1]:
                    idx = ranks[iq, r]
                    mimg = dbimgs[idx]
                    if mimg != qimg:
                        fid.write('{:s}, {:s}, {:g}\n'.format(qimg, mimg, _scores[iq, r]))
                        k += 1
                    r += 1
    logger.debug('\tdone in {:g} seconds'.format(time.time()-t0))


def compute_pairs_command_line():
    parser = argparse.ArgumentParser(description='Compute pairs using HOW or FIRe',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet',
                                  action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('--parameters', default='eval_fire.yml',
                        type=str, help="path to a yaml file that contains parameters.")
    parser.add_argument('--model',  default="fire", choices=['how', 'fire'], type=str, help='type of model')
    parser.add_argument("--model-load", "-ml",  default=None, type=str,
                        help='checkpoint path (overwrites demo_eval.net_path)')
    parser.add_argument('--data-folder', default=None,
                        help='overwrite yml demo_eval.data_folder with this.')
    parser.add_argument('-c', '--codebook-cache-path', default=None, type=str, help='path to store the codebook')
    parser.add_argument('-ivf', '--ivf-cache-path', default=None, type=str, help='path to store the ivf database')
    parser.add_argument('--block-size', default=5000, type=int,
                        help=('max number of local features to store at once for ivf building and querying'))
    parser.add_argument('--map', required=True, help='input path to kapture map root directory.')
    parser.add_argument('--query', default=None, help='input path to kapture query root directory.')
    parser.add_argument('-o', '--output', required=True, help='output path to pairs text file')
    parser.add_argument('--topk', default=50, type=int, help='max number of images per query in output pairs')

    args = parser.parse_args()
    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    logger.debug('kapture_compute_pairs.py \\\n' + ''.join(['\n\t{:13} = {}'.format(k, v)
                                                            for k, v in vars(args).items()]))
    compute_pairs(args.map, args.query,
                  args.model, args.parameters, args.model_load, args.data_folder,
                  args.topk, args.block_size, args.output,
                  args.codebook_cache_path, args.ivf_cache_path)


if __name__ == "__main__":
    compute_pairs_command_line()
