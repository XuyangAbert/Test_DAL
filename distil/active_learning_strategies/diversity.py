# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:53:01 2022

@author: Xuyang
"""

import torch
from math import exp
import numpy as np

from torch.utils.data import dataset, Subset
from .strategy import Strategy
from .margin_sampling import MarginSampling
# from .test_utils import MyLabeledDataset, MyUnlabeledDataset
class Diversity(Strategy):
  def __init__(self, labeled_dataset, unlabeled_dataset, net, nclass, args={}):

    self.strategy = MarginSampling(labeled_dataset,unlabeled_dataset,net,nclass, args)
    super(Diversity,self).__init__(labeled_dataset, unlabeled_dataset, net, nclass, args)

  def dist_cal(self,unlabeled_embeddings):
    unlabeled_embeddings = unlabeled_embeddings.to(self.device)
    # print(unlabeled_embeddings.shape)
    # m = unlabeled_embeddings.shape[0]
    dist_mat = torch.cdist(unlabeled_embeddings,unlabeled_embeddings,p=2)
    return dist_mat
  
  def knei_dist(self,interd,fetch):
    num_nei = 5
    knei_dist = []
    for i in range(interd.shape[0]):
      temp_dist = torch.sort(interd[i][:]).values
      knei_dist.append(torch.sum(temp_dist[:num_nei])/num_nei)
    dth = 0.1*torch.sum(torch.tensor(knei_dist)) / len(knei_dist)
    return dth
    
  def acquire_scores(self, interd):
    priority = []
    beta = torch.mean(interd)
    for i in range(interd.shape[0]):
      priority.append(torch.sum(torch.exp(interd[i][:]/beta)))
    return torch.tensor(priority)
  
  def acquire_scores2(self,unlabeled_batch):
    
    scores = torch.exp(self.strategy.acquire_scores(unlabeled_batch))
    return scores

  def select(self, fetchsize):
    embedding_unlabeled = self.get_embedding(self.unlabeled_dataset)
    # priority1 = self.acquire_scores2(self.unlabeled_dataset)
    bs = 10000
    idx = []
    nb = round(embedding_unlabeled.shape[0]/bs)
    for b in range(nb):
      embedding_unlabeled_batch = embedding_unlabeled[b*bs:(b+1)*bs][:]
      buffered_stream = Subset(self.unlabeled_dataset,list(range(b*bs,min(len(self.unlabeled_dataset),b*bs+bs))))
      # buffered_stream = self.unlabeled_dataset[b*bs:(b+1)*bs]
      interd = self.dist_cal(embedding_unlabeled_batch)
      dth = self.knei_dist(interd, round(fetchsize/nb))
      # print(dth)
      # priority = self.acquire_scores(interd)
      priority = self.acquire_scores2(buffered_stream)
      # priority = priority1[b*bs:(b+1)*bs]
      # print(priority)
      for i in range(round(fetchsize/nb)):
        top_idx = torch.argmax(priority).item()
        idx.append(top_idx)
        neighbordist = interd[top_idx][:]
        neighboridx = torch.where(neighbordist <= dth)[0]
        # len(neighboridx)
        priority[top_idx] = priority[top_idx] / (200 + 200*torch.sum(priority[neighboridx]))  
        priority[neighboridx] = priority[neighboridx] / (200 + 200*torch.sum(priority[neighboridx]))
    # print('Number of quried samples: ',len(torch.unique(torch.tensor(idx))))
    return torch.tensor(idx)
