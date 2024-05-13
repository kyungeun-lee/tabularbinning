import torch, torchvision
import numpy as np

def irevnet(img_size):
    from libs.irevnet import iRevNet
    nBlocks = [6, 16, 72, 6]
    nStrides = [2, 2, 2, 2]
    nChannels = [24, 96, 384, 1536]
    init_ds = 2

    model = iRevNet(nBlocks=nBlocks, nStrides=nStrides,
                    nChannels=nChannels, nClasses=1000, init_ds=init_ds,
                    dropout_rate=0., affineBN=True, in_shape=img_size)
    # model.linear = torch.nn.Identity()
    model = torch.nn.DataParallel(model).cuda()
    return model

def realnvp(img_size):
    import libs.flows as fnn

    modules = []
    num_inputs = np.prod(img_size)

    num_blocks = 5  # number of invertible blocks
    num_hidden = 512
    num_cond_inputs = None

    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(torch.device("cuda")).float()

    for _ in range(num_blocks):
        modules += [
            fnn.CouplingLayer(
                num_inputs, num_hidden, mask, num_cond_inputs,
                s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs)
        ]
        mask = 1 - mask

    model = fnn.FlowSequential(*modules)

    return model

def maf(img_size):
    import libs.flows as fnn

    modules = []
    num_blocks = 5  # number of invertible blocks
    num_hidden = 512
    num_cond_inputs = None

    for _ in range(num_blocks):
        num_inputs = np.prod(img_size)
        modules += [
            fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act="relu"),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]
    model = fnn.FlowSequential(*modules)

    return model


def pretrained_resnet(img_size):
    enc_model = torchvision.models.resnet50(pretrained=False, num_classes=1024).to("cuda")
    weights = torch.load("libs/resnet50_mnist_patch.1.2.5.pt")
    enc_model.load_state_dict(weights["model"], strict=False)
    enc_model.fc = torch.nn.Identity()
    return enc_model.eval()