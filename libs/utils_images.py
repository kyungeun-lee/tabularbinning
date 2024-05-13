import torchvision, torch
import numpy as np

def image_subset(dataname, subclass_list, grayscale=False):
    if dataname == "mnist":
        transform = [torchvision.transforms.Resize(size=(28, 28),
                                                   interpolation=torchvision.transforms.InterpolationMode.BICUBIC)]
        if grayscale:
            transform += [torchvision.transforms.Grayscale()]
        transform += [torchvision.transforms.ToTensor()]
        mnist = torchvision.datasets.MNIST(
            root="mnist", train=True, download=True,
            transform=torchvision.transforms.Compose(transform))
        len_data = len(mnist)

        idx_per_digit = dict()
        for digit in subclass_list:
            idx = mnist.train_labels == digit
            idx_per_digit.update({digit: np.arange(len_data)[idx]})

        return mnist, idx_per_digit

    elif dataname == "cifar10":
        transform = [torchvision.transforms.Resize(size=(32, 32),
                                                   interpolation=torchvision.transforms.InterpolationMode.BICUBIC)]
        if grayscale:
            transform += [torchvision.transforms.Grayscale()]
        transform += [torchvision.transforms.ToTensor()]
        cifar = torchvision.datasets.CIFAR10(
            root="cifar10", train=True, download=True,
            transform=torchvision.transforms.Compose(transform))

        idx_per_image = dict()
        for c in subclass_list:
            idx = [t == c for t in cifar.targets]
            idx_per_image.update({c: np.where(idx)[0]})

        return cifar, idx_per_image

    elif dataname == "cifar100":
        transform = [torchvision.transforms.Resize(size=(32, 32),
                                                   interpolation=torchvision.transforms.InterpolationMode.BICUBIC)]
        if grayscale:
            transform += [torchvision.transforms.Grayscale()]
        transform += [torchvision.transforms.ToTensor()]
        cifar = torchvision.datasets.CIFAR100(
            root="cifar100", train=True, download=True,
            transform=torchvision.transforms.Compose(transform))

        idx_per_image = dict()
        for c in subclass_list:
            idx = [t == c for t in cifar.targets]
            idx_per_image.update({c: np.where(idx)[0]})

        return cifar, idx_per_image
    
    
to_image = torchvision.transforms.ToPILImage()
def get_image(img_size, images, idx_dict, n_patch, bsc_p=0, return_label=False):
    idx = torch.ones(size=n_patch) * 0.5
    idx = torch.bernoulli(idx)
    bsc_idx = torch.ones(size=n_patch) * bsc_p
    bsc_idx = torch.bernoulli(bsc_idx)
    idx_2 = torch.abs(idx - bsc_idx)

    im1, im2 = [], []
    for p1 in range(n_patch[0]):
        im1_1, im2_1 = [], []
        for p2 in range(n_patch[1]):
            im1_2, im2_2 = [], []
            for p3 in range(n_patch[2]):
                i = int(idx[p1, p2, p3])
                j = int(idx_2[p1, p2, p3])

                i = list(idx_dict.keys())[i]
                j = list(idx_dict.keys())[j]
                img1 = np.random.choice(idx_dict[i])
                img2 = np.random.choice(idx_dict[j])
                while img2 == img1:
                    img2 = np.random.choice(idx_dict[j])

                im1_2.append(images[img1][0])
                im2_2.append(images[img2][0])
            im1_1.append(torch.cat(im1_2, dim=-1))
            im2_1.append(torch.cat(im2_2, dim=-1))
        im1.append(torch.cat(im1_1, dim=1))
        im2.append(torch.cat(im2_1, dim=1))

    im1 = to_image(torch.cat(im1))
    im2 = to_image(torch.cat(im2))
    resize = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            size=(img_size, img_size),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.ToTensor()])
    if return_label:
        return resize(im1), resize(im2), idx, idx_2
    else:
        return resize(im1), resize(im2)
    
def image_batch(img_size, images, idx_dict, n_patch, bsc_p=0, batch_size=64, return_label=False):
    image1, image2 = [], []
    if return_label:
        label1, label2 = [], []
    for b in range(batch_size):
        if return_label:
            x1, x2, y1, y2 = get_image(img_size, images, idx_dict, n_patch, bsc_p=bsc_p, return_label=True)

            y1 = y1.numpy().reshape([-1]).tolist()
            y1 = [int(y) for y in y1]

            y2 = y2.numpy().reshape([-1]).tolist()
            y2 = [int(y) for y in y2]

            label1.append(int("".join(map(str, y1)), 2))
            label2.append(int("".join(map(str, y2)), 2))
        else:
            x1, x2 = get_image(img_size, images, idx_dict, n_patch, bsc_p=bsc_p)
        image1.append(x1[None, :, :, :])
        image2.append(x2[None, :, :, :])
    if return_label:
        return torch.cat(image1), torch.cat(image2), label1, label2
    else:
        return torch.cat(image1), torch.cat(image2)



background, _ = image_subset("cifar10", np.arange(10), grayscale=False)
def apply_background(img_size, batch_size, x1, x2, eta, output_channels=1):
    bg_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(
            size=(img_size, img_size),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.ToTensor()])

    bg_idx = np.random.choice(len(background), 2 * batch_size)
    bg_batch = []
    for i in bg_idx:
        bg_batch.append(bg_transform(background.data[i])[None, :, :, :])
    bg_batch = torch.cat(bg_batch)

    z1 = torch.clip(x1 + bg_batch[:batch_size] * eta, 0, 1)
    z2 = torch.clip(x2 + bg_batch[batch_size:] * eta, 0, 1)
    
    if output_channels == 1:
        return (z1.mean(1, keepdims=True), z2.mean(1, keepdims=True)) #(z1, z2)
    else:
        return (z1, z2)