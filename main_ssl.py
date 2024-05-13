import torch, os, zero, torchvision, datetime, time, yaml
from libs.utils import *
from libs.transform import *
from libs.model import TURLModel, CosineAnnealingLR_Warmup, lossfunc

def main(gpu_id, dataname, savepath,
         encodername, decodername, predictorname,
         transform_params, objectives, binning_params, n_decoder, mlpwidth, mlpdepth,
         seed=123456, num_epochs=1000):

    log = logging.getLogger()
    log.setLevel(logging.INFO)
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    file_handler = logging.FileHandler(os.path.join(savepath, 'logs.log'))
    log.addHandler(file_handler)
    log.addHandler(TqdmLoggingHandler())
    log.info("Results will be saved at.. %s" %savepath)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    zero.improve_reproducibility(seed=seed)
    env_info = '{0}:{1}'.format(os.uname().nodename, gpu_id)
    log.info(env_info)

    transform_func = []
    if transform_params is not None:
        for (tfunc, tparams) in transform_params.items():
            if isinstance(tfunc, str):
                log.info("Augmentation: %s" %tfunc)
                tfunc = eval(tfunc)
            transform_func.append(tfunc(**tparams))
    transform_func = torchvision.transforms.Compose(transform_func)

    (train_dataset, train_loader), (val_dataset, val_loader), (cat_features, cat_cardinalities, num_features), batch_size = data_loader(
        dataname, device, transform_func=transform_func, num_bins=binning_params.get("num_bins", 2), mode="pretrain")
    input_dim = len(cat_features) + len(num_features)

    encoder_params = {"modelname": encodername, "d_in": input_dim, "d_out": mlpwidth, "d_layers": [mlpwidth]*mlpdepth}
    decoder_params = {"modelname": decodername, "d_in": mlpwidth, "d_out": input_dim, "d_layers": [mlpwidth]*mlpdepth}
    predictor_params = {"modelname": predictorname}
    log.info(f'encoder: {encoder_params}')
    log.info(f'decoder: {decoder_params}')
    log.info(f'predictor: {predictor_params}')

    model = TURLModel(encoder_params, decoder_params, predictor_params, n_decoder=n_decoder)
    model.to(device)

    optimizer = model.make_optimizer({"lr": 1e-4, "weight_decay": 1e-5})
    scheduler_ops = dict({"warmup_epochs": 0, "T_max": num_epochs, "iter_per_epoch": train_dataset.__len__()//batch_size})
    scheduler = CosineAnnealingLR_Warmup(optimizer, base_lr=1e-4, **scheduler_ops)

    optimizer.zero_grad(); optimizer.step()

    steps_per_epoch = train_dataset.__len__() // batch_size
    pbar = tqdm(range(1, num_epochs + 1))
    
    st = time.time()
    for epoch in pbar:
        pbar.set_description("EPOCH: %i (GPU: %s, %s)" %(epoch, env_info, datetime.datetime.now()))

        sample_idx = np.arange(train_dataset.__len__())
        np.random.shuffle(sample_idx)

        for iteration in range(steps_per_epoch):
            model.train(); optimizer.zero_grad()

            batch_idx = sample_idx[(batch_size*iteration):(batch_size*(iteration+1))]
            x_raw_batch, x_bin_batch = train_dataset[batch_idx]
            x_batch = x_raw_batch['image']

            x_cat = x_batch[:, cat_features].type(torch.int) if len(cat_features) > 0 else None            
            x_num = x_batch[:, num_features] if len(num_features) > 0 else torch.tensor([], device=device)

            z1, z2, z3 = model(x_num, x_cat)

            loss = 0.; decoder_idx = 0
            for (objfunc, objwt) in objectives.items():
                if n_decoder > 1:
                    loss += objwt * lossfunc(objfunc, z2[decoder_idx], z3, x_batch, x_raw_batch['mask'], x_bin_batch)
                    decoder_idx += 1
                else:
                    loss += objwt * lossfunc(objfunc, z2, z3, x_batch, x_raw_batch['mask'], x_bin_batch)

            loss.backward(); optimizer.step(); scheduler.step()

        pbar.set_postfix_str(f'DATA: {dataname}, Tr loss: {loss:.5f}')

        if epoch % 100 == 0:
            log.info(f'EPOCH: {epoch:2d} ...Elapsed: {time.time()-st:.3f} secs ... LOSS: {loss:.5f}')

    torch.save(model.state_dict(), os.path.join(savepath, f'model_encoder_ep{epoch}.pt'))


if __name__ == "__main__":
    
    with open("config.yaml", 'r') as config:
        args = yaml.safe_load(config)
    
    main(**args)