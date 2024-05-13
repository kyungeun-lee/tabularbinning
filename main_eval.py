import torch, os, zero, torchvision, sklearn, scipy, argparse
import torch.nn.functional as F
from libs.utils import *
from libs.transform import *
from libs.model import TURLModel, CosineAnnealingLR_Warmup

def main(gpu_id, dataname, encodername, 
         mlpwidth, mlpdepth, wt_path, finetuning=False,
         num_epochs=100, seed=123456):
    
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    
    if finetuning:
        savepath = os.path.join(wt_path.split("model_encoder_ep1000.pt")[0], "finetuning")
    else:
        savepath = os.path.join(wt_path.split("model_encoder_ep1000.pt")[0], "validation")
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    file_handler = logging.FileHandler(os.path.join(savepath, 'logs.log'))
    log.addHandler(file_handler)
    log.addHandler(TqdmLoggingHandler())
    log.info(f'Results will be saved at.. {savepath}')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    zero.improve_reproducibility(seed=seed)
    env_info = '{0}:{1}'.format(os.uname().nodename, gpu_id)
    log.info(env_info)

    (train_dataset, train_loader), (val_dataset, val_loader), (test_dataset, test_loader), (cat_features, cat_cardinalities, num_features), batch_size, (ydim, tasktype, y_std) = data_loader(
        dataname, device, transform_func=None, mode="validation")
    input_dim = len(cat_features) + len(num_features)

    encoder_params = {"modelname": encodername, "d_in": input_dim, "d_out": mlpwidth, "d_layers": [mlpwidth]*mlpdepth}
    decoder_params = {"modelname": "identity"}
    predictor_params = {"modelname": "identity"}
    log.info(f'encoder: {encoder_params}')
    log.info(f'decoder: {decoder_params}')
    log.info(f'predictor: {predictor_params}')

    model = TURLModel(encoder_params, decoder_params, predictor_params, head=True, ydim=ydim)

    print("!! HEAD !!")
    print(model.head)
    
    weights = torch.load(wt_path)
    model.load_state_dict(weights, strict=False)
    print(" ##### Weights are loaded")
    model.to(device)
    
    scheduler_ops = dict({"warmup_epochs": 0, "T_max": num_epochs, "iter_per_epoch": train_dataset.__len__()//batch_size})
    if finetuning:
        optimizer = model.make_optimizer({"lr": 0.001, "weight_decay": 1e-5})
        scheduler = CosineAnnealingLR_Warmup(optimizer, base_lr=0.001, **scheduler_ops)
    else:
        for name, param in model.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.01, weight_decay=0)
        scheduler = CosineAnnealingLR_Warmup(optimizer, base_lr=0.01, **scheduler_ops)

    optimizer.zero_grad(); optimizer.step()

    loss_fn = (
        F.binary_cross_entropy_with_logits
        if tasktype == 'binclass'
        else F.cross_entropy
        if tasktype == 'multiclass'
        else F.mse_loss
    )

    def evaluate(data, dataloader):
        with torch.no_grad():
            model.eval()
            target = []; prediction = []
            for iteration, batch_idx in enumerate(dataloader):
                x_batch, y_batch = data[batch_idx]

                x_cat = x_batch[:, cat_features].type(torch.int) if len(cat_features) > 0 else None            
                x_num = x_batch[:, num_features] if len(num_features) > 0 else torch.tensor([], device=device)
                _, yhat = model(x_num, x_cat)

                target.append(y_batch)
                prediction.append(yhat)

            target = torch.cat(target, axis=0)
            prediction = torch.cat(prediction, axis=0)
            if tasktype == 'binclass':
                prediction = torch.sigmoid(prediction).round()
                score = torch.sum(target == prediction) / torch.numel(target)
                prediction = torch.argmax(prediction, dim=1)
                target = torch.argmax(target, dim=1)
            elif tasktype == 'multiclass':      
                prediction = torch.argmax(prediction, dim=1)
                target = torch.argmax(target, dim=1)
                score = torch.sum(target == prediction) / torch.numel(target)
            else:
                assert tasktype == 'regression'
                score = ((prediction * y_std - target * y_std)**2).mean()**0.5

        return score

    steps_per_epoch = train_dataset.__len__() // batch_size
    pbar = tqdm(range(1, num_epochs + 1))
    for epoch in pbar:
        pbar.set_description("EPOCH: %i" %epoch)

        sample_idx = np.arange(train_dataset.__len__())
        np.random.shuffle(sample_idx)
        for iteration in range(steps_per_epoch):
            model.train(); optimizer.zero_grad()
            if not finetuning:
                model.encoder.eval()

            batch_idx = sample_idx[(batch_size*iteration):(batch_size*(iteration+1))]
            x_batch, y_batch = train_dataset[batch_idx]
            x_cat = x_batch[:, cat_features] if len(cat_features) > 0 else None
            x_num = x_batch[:, num_features] if len(num_features) > 0 else torch.tensor([], device=device)

            _, yhat_batch = model(x_num, x_cat)
            loss = loss_fn(yhat_batch, y_batch.to(device))
            loss.backward(); optimizer.step(); scheduler.step()

        train_score = evaluate(train_dataset, train_loader)
        val_score = evaluate(val_dataset, val_loader)
        test_score = evaluate(test_dataset, test_loader)

        lr = scheduler.optimizer.param_groups[0]['lr']

        pbar.set_postfix_str(f'Data: {dataname}, LR: {lr:.10f}, Tr loss: {loss:.5f}, Tr score: {train_score:.5f}, Val score: {val_score:.5f}, Test score: {test_score:.5f}')
        log.info(f'EPOCH: {epoch:2d} ... SCORE: train {train_score:.5f}, val {val_score:.5f}, test {test_score:.5f}')
        

    print("============================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #Setup
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataname", type=str, default="CH")
    parser.add_argument("--finetuning", type=int, default=1)
    
    parser.add_argument("--encodername", type=str, default="mlp")
    parser.add_argument("--mlpwidth", type=int, default=128)
    parser.add_argument("--mlpdepth", type=int, default=1)
    
    parser.add_argument("--wt_path", type=str, default="results/CH/model_encoder_ep1000.pt")
        
    args = parser.parse_args()
    
    main(gpu_id=args.gpu_id, dataname=args.dataname, 
         encodername=args.encodername, finetuning=bool(args.finetuning),
         mlpwidth=args.mlpwidth, mlpdepth=args.mlpdepth, wt_path=args.wt_path)