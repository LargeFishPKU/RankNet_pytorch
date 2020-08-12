import torch
import os
import numpy as np
import random

class RankNet_Dataset:
    def __init__(self, id2voc_file, id_cooccur_file, id_value_file, batch_size, bins_number):
        self.ids = []
        with open(id2voc_file, 'r') as f:
            for line in f:
                line = line.rstrip("\r\n")
                id, name = line.split(' ')
                id = int(id)
                self.ids.append(id)
        self.word_number = len(self.ids)
        self.id_cooccur = np.memmap(id_cooccur_file, dtype = 'int32', mode = 'r', shape = (self.word_number, self.word_number + 1))
        self.id_value = np.memmap(id_value_file, dtype = 'float32', mode = 'r', shape = (self.word_number, self.word_number))
        self.batch_size = batch_size
        self.bins_number = bins_number

    def get_word_number(self):
        return self.word_number

    def __getitem__(self, i):
        context_id = self.ids[i]
        all_targets_number = self.id_cooccur[context_id][-1]
        all_target_ids = np.array(self.id_cooccur[context_id][:all_targets_number])
        all_target_values = np.array(self.id_value[context_id][:all_targets_number])
        increase_order_index = np.argsort(all_target_values)
        decrease_order_index = increase_order_index[::-1]
        # order the whole array
        ordered_ids = all_target_ids[decrease_order_index]
        ordered_values = all_target_values[decrease_order_index]
        # generate labels for all target words
        all_labels = self.generate_labels(all_targets_number, self.bins_number)
        # sample target words, the number is batch_size
        sample_index = random.sample(range(0, all_targets_number), self.batch_size)
        sample_ids = ordered_ids[sample_index]
        # print("all_targets_number is {}".format(all_targets_number))
        # print("len of index is {}".format(len(sample_index)))
        # print("len of ordered_ids is {}".format(len(ordered_ids)))
        # print("len of all labels is {}".format(len(all_labels)))
        sample_labels = all_labels[sample_index]

        context_id = torch.tensor(context_id)
        target_ids = torch.tensor(sample_ids)
        labels = torch.tensor(sample_labels)
        return context_id, target_ids, labels


    def __len__(self):
        return self.word_number

    def generate_labels(self, target_number, bins_number):
        labels = []
        k = dict()
        for i in range(bins_number):
            k_temp =( (i+1.0) / bins_number) * target_number - 1
            k_temp = int(k_temp + 0.5)
            k[i+1] = k_temp
        for i in range(target_number):
            for level, value in k.items():
                if i <= value:
                    labels.append(level)
                    break

        labels = np.array(labels)
        return labels
