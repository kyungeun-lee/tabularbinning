import torch, rtdl
import numpy as np
import sklearn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Type, Union
from rtdl.modules import (
    _INTERNAL_ERROR_MESSAGE,
    CategoricalFeatureTokenizer,
    NumericalFeatureTokenizer,
    _TokenInitialization,
)

class TURLModel(torch.nn.Module):
    def __init__(self, 
                 encoder_params, decoder_params, predictor_params, 
                 n_decoder=1, n_predictor=1, device='cuda', seed=123456,
                 head=False, ydim=0):
        super(TURLModel, self).__init__()
        
        torch.manual_seed(seed)
        self.encoder = build_model(**encoder_params)
        self.n_decoder = n_decoder
        if n_decoder > 1:
            self.decoder = []
            for i in range(n_decoder):
                self.decoder.append(build_model(**decoder_params).to(device))
        else:
            self.decoder=build_model(**decoder_params)
            
        self.n_predictor = n_predictor
        if n_predictor > 1:
            self.predictor = []
            for i in range(n_predictor):
                self.predictor.append(build_model(**predictor_params).to(device))
        else:
            self.predictor = build_model(**predictor_params)
            
        if head:
            self.val = True
            self.head = torch.nn.Linear(encoder_params.get("d_out"), ydim)
            self.head.weight.data.normal_(mean=0.0, std=0.010)
            self.head.bias.data.zero_()
        else:
            self.val = False
        
    def forward(self, x_num, x_cat):
        if x_cat is None:
            z1 = self.encoder(x_num)
        else:
            z1 = self.encoder(torch.cat((x_num, x_cat), dim=1))
        
        if self.n_decoder > 1:
            z2 = []
            for hat in self.decoder:
                z2.append(hat(z1))
        else:
            z2 = self.decoder(z1)
        
        if self.n_predictor > 1:
            z3 = []
            for pred in self.predictor:
                z3.append(pred(z2))
        else:
            z3 = self.predictor(z2)
        
        if self.val:
            yhat = self.head(z1)
            return (z1, yhat)
        else:
            return (z1, z2, z3)
    
    def optimization_param_groups(self):
        """The replacement for :code:`.parameters()` when creating optimizers.

        Example::
            optimizer = AdamW(
                model.optimization_param_groups(), lr=1e-4, weight_decay=1e-5
            )
        """
        no_wd_names = ["feature_tokenizer", "normalization", "bias", "pos_embedding", "bn"]

        def needs_wd(name):
            return all(x not in name for x in no_wd_names)

        return [
            {"params": [v for k, v in self.named_parameters() if needs_wd(k)]},
            {
                "params": [v for k, v in self.named_parameters() if not needs_wd(k)],
                "weight_decay": 0.0,
            },
        ]
    
    def make_optimizer(self, opt_params) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            self.optimization_param_groups(),
            **opt_params
        )

def build_model(modelname="mlp", 
                d_in=512, d_out=512,
                ## MLP Parameters
                d_layers=[512, 512, 512], activation=torch.nn.ReLU, dropout=0.1):
    if modelname == "mlp":
        return rtdl.MLP(d_in=d_in, d_out=d_out, d_layers=d_layers,
                        dropouts=dropout, activation=activation)
    elif modelname == "identity":
        return torch.nn.Identity()
    elif (modelname == "upsampling") & (d_out is not None):
        return UpsamplingCNN(d_out)

class Reshape(torch.nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x, target_shape):
        return x.view(target_shape)

class UpsamplingCNN(torch.nn.Module):
    def __init__(self, num_bins):
        super().__init__()
        self.reshape = Reshape()
        self.conv = nn.Conv2d(1, num_bins, kernel_size=1, padding=0, bias=False)
        
    def forward(self, x):
        x_reshape = self.reshape(x, (x.size(0), 1, x.size(1), 1))
        return self.conv(x_reshape)


class CosineAnnealingLR_Warmup(object):
    def __init__(self, optimizer, warmup_epochs, T_max, iter_per_epoch, base_lr, warmup_lr=1e-6, eta_min=0, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.iter_per_epoch = iter_per_epoch
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.eta_min = eta_min
        self.last_epoch = last_epoch

        self.warmup_iter = self.iter_per_epoch * self.warmup_epochs
        self.cosine_iter = self.iter_per_epoch * (self.T_max - self.warmup_epochs)
        self.current_iter = (self.last_epoch + 1) * self.iter_per_epoch

        self.step()

    def get_current_lr(self):
        if self.current_iter < self.warmup_iter:
            current_lr = (self.base_lr - self.warmup_lr) / self.warmup_iter * self.current_iter + self.warmup_lr
        else:
            current_lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * (self.current_iter-self.warmup_iter) / self.cosine_iter)) / 2
        return current_lr

    def step(self):
        current_lr = self.get_current_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        self.current_iter += 1


def CosineAnnealingParam(warmup_epochs, T_max, iter_per_epoch, current_iter, base_value, 
                         warmup_value=1e-8, eta_min=0):
    warmup_iter = iter_per_epoch * warmup_epochs
    cosine_iter = iter_per_epoch * (T_max - warmup_epochs)
    
    if current_iter < warmup_iter:
        return (base_value - warmup_value) / warmup_iter * current_iter + warmup_value
    else:
        return eta_min + (base_value - eta_min) * (1 + np.cos(np.pi * (current_iter - warmup_iter) / cosine_iter)) / 2

    
def lossfunc(objfunc, z2, z3, x_batch, x_mask, x_bin_batch):
    if objfunc == "clf_mask":
        return F.binary_cross_entropy_with_logits(z2, x_mask.to(torch.float32))
    elif objfunc == "clf_binning":
        batch_loss = 0
        num_bins = z3.size(1)
        for col in range(z3.size(2)):
            batch_loss += torch.nn.CrossEntropyLoss()(z3[:, :, col, 0], torch.nn.functional.one_hot(x_bin_batch[:, col], num_bins).to(torch.float32))
        return batch_loss / z3.size(2)
    elif objfunc == "mse_recon":
        return torch.nn.MSELoss()(z2, x_batch.to(torch.float32))
    elif objfunc == "mse_binning":
        return torch.nn.MSELoss()(z2, x_bin_batch.to(torch.float32))                        
    else:
        raise ValueError("Objective function is not defined! (%s)" %objfunc)
        return 