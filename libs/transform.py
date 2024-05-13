from collections import defaultdict
import torch, random
import numpy as np

class Masking(object):
    def __init__(self, mask_prob, masking_constant):
        self.mask_prob = mask_prob
        if type(masking_constant) == str:
            self.masking_constant = eval(masking_constant)
        else:
            self.masking_constant = masking_constant
    
    def __call__(self, sample):
        img = sample['image']
        mask = np.random.uniform(0, 1, size=img.shape) < self.mask_prob    
        img[mask] = self.masking_constant
        return {'image': img, 'mask': torch.tensor(mask, device='cuda').requires_grad_(requires_grad=False)}

class Shuffling(object):
    def __init__(self, mask_prob):
        self.mask_prob = mask_prob
        self.seed = random.randint(0, 99999)
    
    def __call__(self, sample):
        img = sample['image'].to('cuda')
        mask = np.random.uniform(0, 1, size=img.shape) < self.mask_prob
        mask = torch.tensor(mask, device='cuda')
        
        permuted = torch.empty(size=img.size()).to('cuda')
        for f in range(img.size(1)):
            permuted[:, f] = img[torch.randperm(img.size(0)), f]

        return {'image': img * (1-mask.type(torch.int)) + permuted * mask.type(torch.int), 'mask': mask}

class ToTensor(object):
    def __call__(self, sample):
        if isinstance(sample['image'], np.ndarray):
            return {'image': torch.from_numpy(sample['image']), 'mask': torch.from_numpy(sample['mask'])}
        else:
            return {'image': sample['image'], 'mask': sample['mask']}
