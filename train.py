import argparse
import time
import random
import os
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ParcelDataset
from src.tsvit.TSViTcls import TSViT


parser = argparse.ArgumentParser(description="TSViT baseline for T3S crop mapping")

# model config
parser.add_argument('--img_res', default= 1, type= int,
                    help= "Image size.")
parser.add_argument('--patch_size', default= 1, type=int,
                    help= "Spatial patch size.")
parser.add_argument('--dim', type= int, default= 128)
parser.add_argument('--temporal_depth', type= int, default= 4)
parser.add_argument('--spatial_depth', type= int, default= 0)
parser.add_argument('--heads', type= int, default= 4)
parser.add_argument('--dim_head', type= int, default= 32)
parser.add_argument('--dropout', type= float, default= 0.0)
parser.add_argument('--emb_dropout', type= float, default= 0.0)
parser.add_argument('--pool', type= str, default= 'cls', choices= ['cls', 'mean'])
parser.add_argument('--scale_dim', type= int, default= 4)

# data parameters
parser.add_argument('--num_workers', type= int, default= 4)
parser.add_argument('--cache_dir', type= str, default= 'data/processed')
parser.add_argument('--train_years', type= int, nargs= '+', default= [2019, 2020, 2021, 2022, 2025])
parser.add_argument('--val_years', type= int, nargs= '+', default= [2023])
parser.add_argument('--test_years', type= int, nargs= '+', default= [2024])
parser.add_argument('--acq_type', type= str, default= 'clear', choices= ['clear', 'all'])
parser.add_argument('--bands', type= str, nargs= '+', 
                    default= ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'])
parser.add_argument('--temp_var', type= str, default= 'cgdd', choices= ['cgdd', 'cgdd_seasonal'])
parser.add_argument('--ignore_index', type= int, default= 0)
parser.add_argument('--min_px', type= int, default= 1)
parser.add_argument('--min_purity', type= float, default= 0.0)
parser.add_argument('--drop_overlay', action= 'store_true')
parser.add_argument('--keep_invalid', action= 'store_true')
parser.add_argument('--min_clear_frac', type= float, default= 0.0)
parser.add_argument('--max_seq_len', type= int, default= 24)
#parser.add_argument('--use_gdd_pe', action= 'store_true',
#                    help= "Use z-scored CGDD as temporal PE instead of DOY.")
parser.add_argument('--use_temp_calendar', action= 'store_true')
parser.add_argument('--use_temp_subsampling', action= 'store_true')
parser.add_argument('--use_random_subsampling', action= 'store_true')
#parser.add_argument('--use_fixed_temperature_subsampling', action= 'store_true')
#parser.add_argument('--use_upsampling', action= 'store_true')
parser.add_argument('--no_normalize_timestamps', action= 'store_true')

# experiment parameters
parser.add_argument('--res_dir', type= str, default= 'storage/')
parser.add_argument('--checkpoint_dir', type= str, default= None)
parser.add_argument('--overwrite', action= 'store_true',
                    help= "Ignore existing checkpoints and start fresh.")
parser.add_argument('--resume', action= 'store_true')
parser.add_argument('--checkpoint_steps', type= int, default= 500)
parser.add_argument('--epochs', type= int, default= 15)
parser.add_argument('--batch_size', type= int, default= 512, 
                    help= "Parcels per GPU.")
parser.add_argument('--debug_steps', type= int, default= None)
parser.add_argument('--accumulate_steps', type= int, default= 1)
parser.add_argument('--grad_clip', type= float, default= 1.0)
parser.add_argument('--pct_start', type= float, default= 0.05)
parser.add_argument('--lr', type= float, default= 1e-3)
parser.add_argument('--lr_start', type= float, default= 1e-4)
parser.add_argument('--lr_end', type= float, default= 1e-7)
parser.add_argument('--weight_decay', type= float, default= 0.01)
#parser.add_argument('--lr_reference_batch_size', type=int, default=0,
#                    help='if >0: scale lr by effective_batch / this (the tile script uses 64 tiles); 0 = off')
parser.add_argument('--val_every', type= int, default= 1)
parser.add_argument('--select_metric', type= str, default= 'mIoU', choices= ['OA', 'IoU', 'mIoU', 'mF1'])

# misc parameters
parser.add_argument('--rdm_seed', type= int, default= 6666)
parser.add_argument('--seed', type= int, default= None)
parser.add_argument('--device', type= str, default= 'cuda')
parser.add_argument('--display_step', type= int, default= 200)
parser.add_argument('--eval_only', action= 'store_true', 
                    help= "Skip training, test res_dir/model.pth")

# loss parameters
parser.add_argument('--use_class_balance_loss', action= 'store_true')
parser.add_argument('--beta_class_balance', default= 0.99999, type= float)
parser.add_argument('--w_ce', default= 1.0, type= float)
parser.add_argument('--w_focal', default= 0.0, type= float)
parser.add_argument('--w_dice', default= 0.0, type= float)
parser.add_argument('--w_logcosh_dice', default= 0.0, type= float)
parser.add_argument('--w_kl', default= 0.0, type= float)
parser.add_argument('--focal_gamma', default= 2.0, type= float)
parser.add_argument('--w_tversky', default= 0.0, type= float)
parser.add_argument('--tversky_alpha', default= 0.3, type= float)
parser.add_argument('--tversky_beta', default= 0.7, type= float)


def build_input(batch, device):
    """Builds batched input for TSViT from a dataset batch. Invalid observations are dropped.
    Returns:
        torch.tensor: [B, T, 1, 1, C+1]"""
    mask = batch['mask']
    keep = mask.any(dim= 1)
    if not keep.any():
        return None, None

    band_vals = batch['band_vals'][keep]    # (B, T, C)
    timestamps = batch['timestamps'][keep]  # (B, T)

    x = torch.cat([band_vals, timestamps.unsqueeze(-1)], dim= -1)[:, :, None, None, :]  # (B, T, 1, 1, C+1)
    y = batch['label'][keep]

    return x.to(device, non_blocking= True), y.to(device, non_blocking= True)


def compute_metrics(confusion_matrix, ignore_index):
    """Confusion matrix (rows = true, cols = predicted) -> OA, GIoU (= OA / (2 - OA)), mIoU, mF1.
    mIoU and mF1 average over classes present in the ground truth."""
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis= 0) - tp
    fn = confusion_matrix.sum(axis= 1) - tp
    support = confusion_matrix.sum(axis= 1)

    valid = support > 0
    valid[ignore_index] = False
    if not valid.any():
        return {'OA': 0.0, 'IoU': 0.0, 'mIoU': 0.0, 'mF1': 0.0}
    
    oa = float(tp[valid].sum() / support[valid].sum())
    iou = tp / np.maximum(tp + fp + fn, 1e-8)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-8)
    
    return {'OA': oa, 'IoU': oa / (2 - oa), 'mIoU': float(iou[valid].mean()), 'mF1': float(f1[valid].mean())}


@torch.no_grad()
def evaluate(model, loader, class_weights, args, device, num_classes):
    model.eval()
    confusion_matrix = torch.zeros(num_classes**2, dtype= torch.float64, device= device)
    
    loss, n = torch.zeros((), device= device), 0
    for step, batch in enumerate(loader):
        if args.debug_steps and step >= args.debug_steps:
            break
        
        x, y = build_input(batch, device)
        if x is None:
            continue
        
        logits = model(x)
        loss += F.cross_entropy(logits, y, weight= class_weights, ignore_index= args.ignore_index) * len(y)
        n += len(y)

        preds = logits.argmax(dim= 1)
        idxs = y * num_classes + preds
        
        confusion_matrix += torch.bincount(idxs, minlength= num_classes**2)

    res = compute_metrics(confusion_matrix.view(num_classes, num_classes).cpu().numpy(), args.ignore_index)
    res.update({'loss': float(loss) / max(n, 1), 'n': n})

    return res


def format_res(m):
    return f"loss {m['loss']:.4f} | parcels {m['n']:,} | OA {m['OA']:.2%} mIoU {m['mIoU']:.2%} mF1 {m['mF1']:.2%}"

    
def iterate(model, optimizer, scheduler, loader, class_weights, args, device, epoch, batches_per_epoch):
    model.train()
    optimizer.zero_grad()
    
    loss, n, t0 = torch.zeros((), device= device), 0, time.time()
    for step, batch in enumerate(loader):
        if step >= batches_per_epoch:
            break

        x, y = build_input(batch, device)
        if x is not None:
            logits = model(x)
            batch_loss = F.cross_entropy(logits, y, weight= class_weights, ignore_index= args.ignore_index)
            (batch_loss / args.accumulate_steps).backward()
            loss += batch_loss.detach()
            n += 1

        if (step + 1) % args.accumulate_steps == 0 or step + 1 == batches_per_epoch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % args.display_step == 0:
            print(f'  epoch {epoch + 1} step {step + 1}/{batches_per_epoch} | loss {float(loss) / max(n, 1):.4f} | '
                  f'lr {scheduler.get_last_lr()[0]:.2e} | {(step + 1) * args.batch_size / (time.time() - t0):,.0f} parcels/s', flush= True)
    
    return float(loss) / max(n, 1)


def main():
    args = parser.parse_args()

    # set seed and device
    seed = args.seed if args.seed is not None else args.rdm_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    
    os.makedirs(args.res_dir, exist_ok= True)
    with open(os.path.join(args.res_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent= 2)
    print(f'train {args.train_years} | val {args.val_years} | test {args.test_years}', flush= True)


    # ------ Data ------------------------------------ #
    
    print("Building datasets...", flush= True)
    dataset_config = dict(
        acq_type= args.acq_type, bands= tuple(args.bands),
        temp_var= args.temp_var,
        ignore_index= args.ignore_index,
        min_px= args.min_px, min_purity= args.min_purity,
        drop_overlay= args.drop_overlay, drop_invalid= not args.keep_invalid,
        min_clear_frac= args.min_clear_frac,
        max_seq_len= args.max_seq_len, use_temp_calendar= args.use_temp_calendar,
        use_temp_subsampling= args.use_temp_subsampling,
        use_random_subsampling= args.use_random_subsampling,
        normalize_timestamps= not args.no_normalize_timestamps
    )
    train_ds = ParcelDataset(args.cache_dir, args.train_years, training= True, **dataset_config)
    norm_stats = dict(band_stats= train_ds.band_stats, time_stats= train_ds.time_stats)
    with open(os.path.join(args.res_dir, 'norm_stats.json'), 'w') as f:
        json.dump(norm_stats, f, default= lambda o: o.tolist())
    val_ds  = ParcelDataset(args.cache_dir, args.val_years,  training= False, **norm_stats, **dataset_config)
    test_ds = ParcelDataset(args.cache_dir, args.test_years, training= False, **norm_stats, **dataset_config)
    num_classes = train_ds.num_classes
    print(f'Parcels: train {len(train_ds):,} | val {len(val_ds):,}' + (f' | test {len(test_ds):,}' if test_ds else '')
            + f' | classes {num_classes}', flush= True)

    dataloader_config = dict(
        batch_size= args.batch_size, 
        num_workers= args.num_workers,
        persistent_workers= args.num_workers > 0,
        pin_memory= device.type == 'cuda'
    )
    train_loader = DataLoader(train_ds, shuffle= True,  **dataloader_config)
    val_loader   = DataLoader(val_ds,   shuffle= False, **dataloader_config)
    test_loader  = DataLoader(test_ds,  shuffle= False, **dataloader_config)


    # ------ Model ------------------------------------ #
    
    print("Building model...", flush= True)
    model_config = dict(
        img_res= args.img_res, patch_size= args.patch_size, 
        num_classes= num_classes, max_seq_len= args.max_seq_len,
        dim= args.dim, temporal_depth= args.temporal_depth, spatial_depth= args.spatial_depth,
        heads= args.heads, dim_head= args.dim_head,
        dropout= args.dropout, emb_dropout= args.emb_dropout, pool= args.pool,
        scale_dim= args.scale_dim,
        num_channels= len(args.bands) + 1, 
    )
    model = TSViT(model_config).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}', flush= True)

    class_weights = torch.tensor(train_ds.get_class_weights(args.beta_class_balance), device= device) if args.use_class_balance_loss else None
    
    optimizer = torch.optim.AdamW(model.parameters(), lr= args.lr, weight_decay= args.weight_decay)
    
    batches_per_epoch = min(len(train_loader), args.debug_steps) if args.debug_steps else len(train_loader)
    total_steps = args.epochs * math.ceil(batches_per_epoch / args.accumulate_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr= args.lr, total_steps= total_steps, pct_start= args.pct_start,
        div_factor= args.lr / args.lr_start, final_div_factor= args.lr_start / args.lr_end,
        cycle_momentum= False
    )

    latest_path, best_path = os.path.join(args.res_dir, 'model_latest.pth'), os.path.join(args.res_dir, 'model_best.pth')
    start_epoch, best, history = 0, -1.0, []
    if args.resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location= device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch, best, history = ckpt['epoch'] + 1, ckpt['best'], ckpt['history']
        print(f'resumed from epoch {start_epoch}', flush= True)

    
    # ------ training ------------------------------------ #
    
    if not args.eval_only:
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            train_loss = iterate(model, optimizer, scheduler, train_loader, class_weights, 
                                 args, device, epoch, batches_per_epoch)
            print(f'epoch {epoch + 1}/{args.epochs} | train loss {train_loss:.4f} | {(time.time() - t0) / 60:.1f} min', flush= True)
            if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
                val = evaluate(model, val_loader, class_weights, args, device, num_classes)
                print('  val: ' + format_res(val), flush= True)
                history.append({'epoch': epoch + 1, 'train_loss': train_loss, **val})
                if val[args.select_metric] > best:
                    best = val[args.select_metric]
                    torch.save(model.state_dict(), best_path)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best': best, 'history': history}, latest_path)

    # ---- final evaluation with the best model ----
    model.load_state_dict(torch.load(best_path, map_location= device))
    results = {'history': history, 'best_' + args.select_metric: best}
    results['val'] = evaluate(model, val_loader, class_weights, args, device, num_classes)
    print('best model, val:  ' + format_res(results['val']), flush= True)
    if test_loader is not None:
        results['test'] = evaluate(model, test_loader, class_weights, args, device, num_classes)
        print('best model, test: ' + format_res(results['test']), flush= True)
    with open(os.path.join(args.res_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent= 2)


if __name__ == '__main__':
    main()