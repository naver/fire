# Copyright (C) 2021-2022 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import torch
from torch import nn
import torchvision

from cirtorch.networks import imageretrievalnet

from how import layers
from how.layers import functional as HF

from lit import LocalfeatureIntegrationTransformer

from how.networks.how_net import HOWNet, CORERCF_SIZE

class FIReNet(HOWNet):

    def __init__(self, features, attention, lit, dim_reduction, meta, runtime):
        super().__init__(features, attention, None, dim_reduction, meta, runtime)
        self.lit = lit
        self.return_global = False
        
    def copy_excluding_dim_reduction(self):
        """Return a copy of this network without the dim_reduction layer"""
        meta = {**self.meta, "outputdim": self.meta['backbone_dim']}
        return self.__class__(self.features, self.attention, self.lit, None, meta, self.runtime)

    def copy_with_runtime(self, runtime):
        """Return a copy of this network with a different runtime dict"""
        return self.__class__(self.features, self.attention, self.lit, self.dim_reduction, self.meta, runtime)

    def parameter_groups(self):
        """Return torch parameter groups"""
        layers = [self.features, self.attention, self.smoothing, self.lit]
        parameters = [{'params': x.parameters()} for x in layers if x is not None]
        if self.dim_reduction:
            # Do not update dimensionality reduction layer
            parameters.append({'params': self.dim_reduction.parameters(), 'lr': 0.0})
        return parameters

    def get_superfeatures(self, x, *, scales):
        """
        return a list of tuple (features, attentionmpas) where each is a list containing requested scales
        features is a tensor BxDxNx1
        attentionmaps is a tensor BxNxHxW
        """
        feats = []
        attns = []
        strengths = []
        for s in scales:
            xs = nn.functional.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False)
            o = self.features(xs)
            o, attn = self.lit(o)
            strength = self.attention(o)
            if self.smoothing:
                o = self.smoothing(o)
            if self.dim_reduction:
                o = self.dim_reduction(o)
            feats.append(o)
            attns.append(attn)
            strengths.append(strength)
        return feats, attns, strengths
        
    def forward(self, x):
        if self.return_global:
            return self.forward_global(x, scales=self.runtime['training_scales'])
        return self.get_superfeatures(x, scales=self.runtime['training_scales'])
        
    def forward_global(self, x, *, scales):
        """Return global descriptor"""
        feats, _, strengths = self.get_superfeatures(x, scales=scales)
        return HF.weighted_spoc(feats, strengths)
        
    def forward_local(self, x, *, features_num, scales):
        """Return selected super features"""
        feats, _, strengths = self.get_superfeatures(x, scales=scales)
        return HF.how_select_local(feats, strengths, scales=scales, features_num=features_num)

def init_network(architecture, pretrained, skip_layer, dim_reduction, lit, runtime):
    """Initialize FIRe network
    :param str architecture: Network backbone architecture (e.g. resnet18)
    :param str pretrained: url of the pretrained model (None for using random initialization)
    :param int skip_layer: How many layers of blocks should be skipped (from the end)
    :param dict dim_reduction: Options for the dimensionality reduction layer
    :param dict lit: Options for the lit layer
    :param dict runtime: Runtime options to be stored in the network
    :return FIRe: Initialized network
    """
    # Take convolutional layers as features, always ends with ReLU to make last activations non-negative
    net_in = getattr(torchvision.models, architecture)(pretrained=False) # use trained weights including the LIT module instead 
    if architecture.startswith('alexnet') or architecture.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif architecture.startswith('densenet'):
        features = list(net_in.features.children()) + [nn.ReLU(inplace=True)]
    elif architecture.startswith('squeezenet'):
        features = list(net_in.features.children())
    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))

    if skip_layer > 0:
        features = features[:-skip_layer]
    backbone_dim = imageretrievalnet.OUTPUT_DIM[architecture] // (2 ** skip_layer)

    att_layer = layers.attention.L2Attention()

    lit_layer = LocalfeatureIntegrationTransformer(**lit, input_dim=backbone_dim)

    reduction_layer = None
    if dim_reduction:
        reduction_layer = layers.dim_reduction.ConvDimReduction(**dim_reduction, input_dim=lit['dim'])

    meta = {
        "architecture": architecture,
        "backbone_dim": lit['dim'],
        "outputdim": reduction_layer.out_channels if dim_reduction else lit['dim'],
        "corercf_size": CORERCF_SIZE[architecture] // (2 ** skip_layer),
    }
    net = FIReNet(nn.Sequential(*features), att_layer, lit_layer, reduction_layer, meta, runtime)
    
    if pretrained is not None:
        assert os.path.isfile(pretrained), pretrained
        ckpt = torch.load(pretrained, map_location='cpu')
        missing, unexpected = net.load_state_dict(ckpt['state_dict'], strict=False)
        assert all(['dim_reduction' in a for a in missing]), "Loading did not go well"
        assert all(['fc' in a for a in unexpected]), "Loading did not go well"
    return net
