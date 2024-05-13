import os, glob, torch, torchvision
import numpy as np

to_image = torchvision.transforms.ToPILImage()
def get_text_image_pair(image_dataset, image_idx_dict, image_size,
                        text_dataset, text_idx_dict,
                        image_patch=(1, 2, 5), text_size=10, bsc_p=0, return_label=False):

    assert np.prod(image_patch) == text_size
    idx = torch.ones(size=image_patch) * 0.5
    idx = torch.bernoulli(idx)
    bsc_idx = torch.ones(size=image_patch) * bsc_p
    bsc_idx = torch.bernoulli(bsc_idx)
    idx_2 = torch.abs(idx - bsc_idx).view(-1)

    im = []
    for p1 in range(image_patch[0]):
        im1 = []
        for p2 in range(image_patch[1]):
            im2 = []
            for p3 in range(image_patch[2]):
                i = int(idx[p1, p2, p3])
                i = list(image_idx_dict.keys())[i]
                img1 = np.random.choice(image_idx_dict[i])

                im2.append(image_dataset[img1][0])
            im1.append(torch.cat(im2, dim=-1))
        im.append(torch.cat(im1, dim=1))

    te = []
    for t in range(text_size):
        j = int(idx_2[t])
        text1 = np.random.choice(np.arange(text_idx_dict[j], text_idx_dict[j + 1]))
        te.append(torch.Tensor(text_dataset[text1][0]))
    texts = torch.cat(te, dim=-1)
    
    if image_size*image_size == texts.size(1):
        im = to_image(torch.cat(im))
        resize = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                size=(image_size, image_size),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor()])
    elif texts.size(1) == 7680:
        im = to_image(torch.cat(im))
        resize = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                size=(80, 96),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor()])
    if return_label:
        return resize(im), texts, idx, idx_2
    else:
        return resize(im), texts


def mixture_batch(image_dataset, image_idx_dict, text_dataset, img_size, n_patches, n_text, bsc_p=0, batch_size=64, return_label=False):

    image1, image2 = [], []
    label1, label2 = [], []
    for b in range(batch_size):
        if return_label:
            x1, x2, y1, y2 = get_text_image_pair(image_dataset, image_idx_dict, img_size,
                                                 text_dataset, text_dataset.counts,
                                                 image_patch=n_patches, text_size=n_text, bsc_p=bsc_p, return_label=True)

            y1 = y1.numpy().reshape([-1]).tolist()
            y1 = [int(y) for y in y1]

            y2 = y2.numpy().reshape([-1]).tolist()
            y2 = [int(y) for y in y2]

            label1.append(int("".join(map(str, y1)), 2))
            label2.append(int("".join(map(str, y2)), 2))
        else:
            x1, x2 = get_text_image_pair(image_dataset, image_idx_dict, img_size,
                                         text_dataset, text_dataset.counts,
                                         image_patch=n_patches, text_size=n_text, bsc_p=bsc_p, return_label=False)

        image1.append(torch.Tensor(x1).view(1, -1))
        image2.append(torch.Tensor(x2).view(1, -1))

    if return_label:
        return torch.cat(image1), torch.cat(image2), label1, label2
    else:
        return torch.cat(image1), torch.cat(image2)