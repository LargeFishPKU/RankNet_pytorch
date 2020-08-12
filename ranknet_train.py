import numpy as np
import torch
import os
import argparse
import torch.optim as optim
import torch.nn.functional as F
from easydict import EasyDict
from ranknet import RankNet
from ranknet_dataset import RankNet_Dataset

data_dir = "/mnt/lustre/yankun/learning_to_rank/data"
id2voc = os.path.join(data_dir, "id2voc.txt")
id_cooccur = os.path.join(data_dir, "id_cooccur.dat")
id_value = os.path.join(data_dir, "id_value.dat")


# training process of
def train_save(batch_size, embed_size, bins_number, iterations, save_embedding_path):
    # get the dataset and the number of words
    embed_size = int(embed_size)
    bins_number = int(bins_number)
    iterations = int(iterations)
    print("prepare the dataset ...")
    dataset = RankNet_Dataset(id2voc, id_cooccur, id_value, batch_size, bins_number)
    word_number = dataset.get_word_number()
    print("the number of words is {}".format(word_number))
    print("done")
    # get the ranknet model
    print("get ranknet model ...", flush=True)
    model = RankNet(word_number, embed_size, bins_number)
    model = model.cuda()
    print("done")

    # get the optimization
    opt = optim.Adam(model.parameters())

    # train the model
    print("training ...")
    dataload = torch.utils.data.DataLoader(dataset, num_workers=12, pin_memory=True)

    for i in range(iterations):
        print("{} / {}".format(i, iterations), flush=True)
        len_dataload = len(dataload)
        avg_loss = 0.0
        for j, (context_id, target_ids, labels) in enumerate(dataload):
            opt.zero_grad()
            loss = model.forward_loss(context_id, target_ids, labels)
            loss.backward()
            opt.step()
            avg_loss = avg_loss + loss.data.item()
            if j % 1000 == 0:
                print("sub: {} / {}, loss : {:.4f}".format(j, len_dataload, avg_loss / (j + 1)), flush = True)

    # save embeddings:
    print("training process is done")
    print("start saving embeddings ...")
    if not os.path.isdir(save_embedding_path):
        os.makedirs(save_embedding_path)
    embedings = model.get_embeddings()

    # save the data as the numpy form:
    filename = "ranknet_embeddings" + str(embed_size) + "_b_" + str(bins_number) + "_i_" + str(iterations)
    file_path = os.path.join(save_embedding_path, filename + '.npy')
    np.save(file_path, embedings)
    # save the data as the txt form:
    file_path = os.path.join(save_embedding_path, filename + '.txt')
    with open(file_path, 'w') as f:
        number_word = len(embedings)
        for i in range(number_word):
            line_i = embedings[i]
            temp = ""
            for j, value in enumerate(line_i):
                if j == 0:
                    temp = temp + str(value)
                else:
                    temp = temp + ' ' + str(value)
            temp = temp + '\n'
            f.write(temp)
    print("done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'pranking algorithm')
    parser.add_argument('--batch_size', default = 200)
    parser.add_argument('--embed_size', default = 300)
    parser.add_argument('--bins_number', default = 10)
    parser.add_argument('--iterations', default = 50)
    parser.add_argument('--save_embedding_path', default = "/mnt/lustre/yankun/learning_to_rank/embedding")

    args = parser.parse_args()
    # import pdb; pdb.set_trace()

    # params = EasyDict(args)
    train_save(args.batch_size, args.embed_size, args.bins_number, args.iterations, args.save_embedding_path)
    # import pdb; pdb.set_trace()
