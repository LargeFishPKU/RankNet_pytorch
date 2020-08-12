import numpy as np
import torch
import os
import torch.nn as nn
from torch.nn.init import xavier_normal


class RankNet(nn.Module):
    def __init__(self, word_number, embed_size, bins_number):
        super(RankNet, self).__init__()

        self.word_number = word_number
        self.embed_size = embed_size
        self.bins_number = bins_number

        # for context words and target words
        self.in_embed = nn.Embedding(self.word_number, self.embed_size)
        self.layer1 = nn.Sequential(
            nn.Linear(self.embed_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
        )
        self.layer2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = self.layer2(x)
        return x

    def forward_loss(self, context_id, target_ids, labels):
        '''
            context_id: (1)
            target_ids: (1, batch_size)
            labels: (1, batch_size)
        '''
        # context_id, target_ids, labels = input
        context_id = context_id.cuda()
        target_ids = target_ids.cuda()
        labels = labels.cuda()

        target_ids = target_ids.view(-1).long()
        labels = labels.view(-1).long()

        context_embedding = self.in_embed(context_id) #(1, embed_size)
        # context_embedding = self.in_embed[context_id] #(1, embed_size)
        target_embeddings = self.in_embed(target_ids) #(batch_size, embed_size)
        # target_embeddings = self.in_embed[target_ids] #(batch_size, embed_size)
        sum_embeddings = context_embedding + target_embeddings #(batch_size, embed_size)

        batch_pred = self.forward(sum_embeddings) #(batch_size, 1)

        batch_size = batch_pred.size(0)

        # sampling pairs from batch, for convenience, let the n_sampling_combs equals to batch_size
        n_sampling_combs = batch_size
        batch_loss = torch.zeros(1).cuda()
        for _ in range(n_sampling_combs):
            i, j = np.random.choice(range(batch_size), 2, replace = False)
            s_i = batch_pred[i]
            s_j = batch_pred[j]
            if labels[i] > labels[j]:
                S_ij = 1
            elif labels[i] == labels[j]:
                S_ij = 0
            else:
                S_ij = -1
            loss = self.pairwise_loss(s_i, s_j, S_ij)
            batch_loss += loss

        batch_loss = batch_loss / n_sampling_combs
        return batch_loss


    def pairwise_loss(self, s_i, s_j, S_ij, sigma = 1):
        C = torch.log1p(torch.exp(-sigma * (s_i - s_j)))
        if S_ij == -1:
            C += sigma * (s_i - s_j)
        elif S_ij == 0:
            C += 0.5 * sigma * (s_i - s_j)
        elif S_ij == 1:
            pass
        else:
            raise ValueError("S_ij: -1/0/1")
        return C


    def get_embeddings(self):
        # return self.in_embed.weight.data.cpu().numpy() + self.out_embed.weight.data.cpu().numpy()
        return self.in_embed.weight.data.cpu().numpy()
        # return self.in_embed.data.cpu().numpy()
