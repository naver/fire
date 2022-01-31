# Copyright (C) 2021-2022 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
from torch import nn
import torch.nn.functional as F

from cirtorch.layers.loss import ContrastiveLoss

class DecorrelationAttentionLoss(nn.Module):

    def __init__(self, weight=1.0):
      super().__init__()
      self.weight = weight

    def forward(self, attention_list):
        """
        attention_list is a list of tensor of size N x H x W where N is the number of attention maps per image
        """
        total_loss = 0.0
        for attn in attention_list:
            assert attn.ndim == 3
            N = attn.size(0)
            attn = attn.view( N, -1)
            attnN = F.normalize(attn, dim=1)
            corr = torch.einsum("rn,sn -> rs", attnN, attnN)
            loss = (corr.sum() - torch.diagonal(corr, dim1=0, dim2=1).sum() ) / (N * (N-1) ) # sum over non-diagonal elements
            total_loss += loss
        return total_loss * self.weight
    
    def __repr__(self):
        return "{:s}(weight={:g})".format(self.__class__.__name__, self.weight)
    
    
def match(query_feat, pos_feat, LoweRatioTh=0.9):
    # first perform reciprocal nn
    dist = torch.cdist(query_feat, pos_feat)
    best1 = torch.argmin(dist, dim=1)
    best2 = torch.argmin(dist, dim=0)
    arange = torch.arange(best2.size(0), device=best2.device)
    reciprocal = best1[best2]==arange
    # check Lowe ratio test
    dist2 = dist.clone()
    dist2[best2,arange] = float('Inf')
    dist2_second2 = torch.argmin(dist2, dim=0)
    ratio1to2 = dist[best2,arange] / dist2_second2
    valid = torch.logical_and(reciprocal, ratio1to2<=LoweRatioTh)
    pindices = torch.where(valid)[0]
    qindices = best2[pindices]
    # keep only the ones with same indices 
    valid = pindices==qindices
    return pindices[valid]
    
    
class SuperfeatureLoss(nn.Module):
    
    def __init__(self, margin=1.1, weight=1.0):
        super().__init__()
        self.weight = weight
        self.criterion = ContrastiveLoss(margin=margin)
        
    def forward(self, superfeatures_list, target):
        """
        superfeatures_list is a list of tensor of size N x D containing the superfeatures for each image
        """
        assert target[0]==-1 and target[1]==1 and torch.all(target[2:]==0), "Only implemented for one tuple where the first element is the query, the second one the positive, and the rest are negatives"
        N = superfeatures_list[0].size(0)
        assert all(s.size(0)==N for s in superfeatures_list[1:]), "All images should have the same number of features"
        query_feat = F.normalize(superfeatures_list[0], dim=1)
        pos_feat = F.normalize(superfeatures_list[1], dim=1)
        neg_feat_list = [F.normalize(neg, dim=1) for neg in superfeatures_list[2:]]
        # perform matching 
        indices = match(query_feat, pos_feat)
        if indices.size(0)==0:
            return torch.sum(query_feat[:1,:1])*0.0 # for having a gradient that depends on the input to avoid torch error when using multiple processes
        # loss
        nneg = len(neg_feat_list)
        target = torch.Tensor( ([-1, 1]+[0]*nneg) * len(indices)).to(dtype=torch.int64, device=indices.device)
        catfeats = torch.cat([query_feat[indices, None, :], pos_feat[indices, None, :]] + \
                             [neg_feat[indices,None,:] for neg_feat in neg_feat_list], dim=1) # take qindices for the negatives
        catfeats = catfeats.view(-1, query_feat.size(1))

        loss = self.criterion(catfeats.T, target.detach())
        return loss * self.weight

    def __repr__(self):
        return "{:s}(margin={:g}, weight={:g})".format(self.__class__.__name__, self.criterion.margin, self.weight)
