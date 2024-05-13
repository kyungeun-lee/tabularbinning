import os, glob, torch
import numpy as np


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataname="imdb.bert-imdb-finetuned",
                 root="dataset", n_sample=None):
        super(TextDataset, self).__init__()

        root = os.path.join(root, dataname)

        self.classes = [0, 1]

        self.data = []
        self.counts = dict()
        self.label = []
        for idx, subclass in enumerate(self.classes):
            file_list = glob.glob(os.path.join(root, str(subclass), '*.npy'))
            file_list.sort()

            if (not n_sample in [None, "None"]):
                if n_sample > 0:
                    file_list = file_list[:n_sample]

            self.counts[idx] = len(self.data)
            self.data += [(filename, idx) for filename in file_list]
            self.label += [idx] * len(file_list)
        self.counts[len(self.classes)] = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename, target_idx = self.data[index]
        data = np.load(filename)

        return data, target_idx

def get_text(dataset, idx_dict, n_text, bsc_p=0, get_label=False):
    idx = torch.ones(size=(n_text,)) * 0.5
    idx = torch.bernoulli(idx)
    bsc_idx = torch.ones(size=(n_text,)) * bsc_p
    bsc_idx = torch.bernoulli(bsc_idx)
    idx_2 = torch.abs(idx - bsc_idx)

    x1, x2 = [], []
    for t in range(n_text):
        i = int(idx[t])
        j = int(idx_2[t])

        text1 = np.random.choice(np.arange(idx_dict[i], idx_dict[i+1]))
        text2 = np.random.choice(np.arange(idx_dict[j], idx_dict[j+1]))

        while text2 == text1:
            text2 = np.random.choice(np.arange(idx_dict[j], idx_dict[j+1]))

        x1.append(torch.Tensor(dataset[text1][0]))
        x2.append(torch.Tensor(dataset[text2][0]))

    if get_label:
        return torch.cat(x1, dim=-1), torch.cat(x2, dim=-1), idx, idx_2
    else:
        return torch.cat(x1, dim=-1), torch.cat(x2, dim=-1)


def text_batch(dataset, n_text, bsc_p=0, batch_size=64, n_sample=None, get_label=False):
    if n_sample is None:
        idx_dict = dataset.counts
    else:
        idx_dict = {0: 0, 1: n_sample, 2: dataset.counts[1]+n_sample}

    image1, image2 = [], []
    label1, label2 = [], []
    for b in range(batch_size):
        if get_label:
            x1, x2, y1, y2 = get_text(dataset, idx_dict, n_text, bsc_p=bsc_p, get_label=True)

            y1 = y1.numpy().reshape([-1]).tolist()
            y1 = [int(y) for y in y1]

            y2 = y2.numpy().reshape([-1]).tolist()
            y2 = [int(y) for y in y2]

            label1.append(int("".join(map(str, y1)), 2))
            label2.append(int("".join(map(str, y2)), 2))
        else:
            x1, x2 = get_text(dataset, idx_dict, n_text, bsc_p=bsc_p, get_label=False)
        image1.append(torch.Tensor(x1).view(1, -1))
        image2.append(torch.Tensor(x2).view(1, -1))
    if get_label:
        return torch.cat(image1), torch.cat(image2), label1, label2
    else:
        return torch.cat(image1), torch.cat(image2)