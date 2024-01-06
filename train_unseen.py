'''
Main program
Code adapted from following paper
"Training Generative Adversarial Networks with Limited Data."
See LICENSES/LICENSE_STYLEGAN2_ADA.txt for original license.
'''

import os
import click
import re
import json
import tempfile
import torch
import dnnlib
import copy

from tqdm import tqdm
from training import training_loop as training_loop
from torch_utils import training_stats
from torch_utils import custom_ops
from training.images_dataset import AllImagesDataset

from configs.default_configs import dataset_configs, path_configs
from configs.args_configs import stage_configs, loss_configs, global_configs


#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

# Set up general settings for both stages
def setup_general_kwargs(
    # General options (not included in desc).
    gpus              = None,           # Number of GPUs: <int>, default = 1 gpu
    snap              = None,           # Snapshot interval: <int>, default = 1 ticks
    seed              = None,           # Random seed: <int>, default = 0
    # Base config.
    cfg               = None,           # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
    gamma             = None,           # Override R1 gamma: <float>
    batch             = None,             # Override batch size: <int>

    # Discriminator augmentation
    aug               = None,           # Augmentation mode: 'ada' (default), 'noaug', 'fixed'
    p                 = None,           # Specify p for 'fixed' (required): <float>
    target            = None,           # Override ADA target for 'ada': <float>, default = depends on aug
    augpipe           = None,           # Augmentation pipeline: 'blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc' (default), ..., 'b
    allow_tf32        = None,           # Allow PyTorch to use TF32 for matmul and convolutions: <bool>, default = False
    nobench           = None,           # Disable cuDNN benchmarking: <bool>, default = False
    workers           = None,           # Override number of DataLoader workers: <int>, default = 3

    # Few-shot args.
    dataset_name      = None,           # Name of dataset
    k_shot            = 1,              # Number of shots
    single_task       = None,           # Run only one task and verbose
    save_ckpt         = False,          # Save checkpoints for each task
    classes           = None,           # Category indexes
    grid_size         = [4, 5]          # Visualization grid size
):
    args = dnnlib.EasyDict()
    # Name of output folder
    desc = dataset_name

    # Set up k
    desc += f'-{k_shot}_shot'
    args.k_shot = k_shot

    # Set stages acccording to datasets
    stage_args = stage_configs[f'{dataset_name}_1'] if k_shot == 1 else stage_configs[dataset_name]

    dataset_params = dataset_configs[dataset_name]
    args.training_set_opts = dnnlib.EasyDict(**dataset_params, dataset_name=dataset_name)

    args.visualization_kwargs = dnnlib.EasyDict(grid_size=grid_size)
    args.visualization_kwargs.n_gen = 128

    # ------------------------------------------
    # General options: gpus, snap, metrics, seed
    # ------------------------------------------

    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    # Setup global opts for few-shot optimization
    args.global_opts = dnnlib.EasyDict(**global_configs)
    args.global_opts.seen_batch_gpu = args.global_opts.seen_batch_size // gpus

    # Whether to perform single task optimization of multiple task evaluation.
    args.global_opts.single_task = single_task

    # Whether to save checkpoint for optimization. Used for single task optimization, not suitable for multiple tasks as the ckpts take up storage.
    args.global_opts.save_ckpt = save_ckpt

    # Whether to show the snapshot of tuning. Used for single task optimization. For multiple tasks, only the final results are stored.
    if snap is None:
        snap = 1
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = 1e8

    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    if cfg is None:
        cfg = 'auto'
    assert isinstance(cfg, str)

    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
        'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
        'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
        'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        spec.ref_gpus = gpus
        # Set up resolution for different resolution. Flowers and AnimalFaces: 128x128, VGGFaces: 64x64.
        res = dataset_params['resolution']
        spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
        spec.mbstd = min(spec.mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32

    # Extent the dimension to n_seen + 1
    args.mapping_kwargs = dnnlib.EasyDict(class_name='training.networks.MappingNetwork', \
        z_dim=512, w_dim=512, c_dim=dataset_params['n_seen']+1, num_layers = spec.map)
    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.TuningLoss', r1_gamma=spec.gamma)

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp

    # Modify some settings for few-shot optimization
    if cfg == 'cifar' or cfg == 'auto':
        args.loss_kwargs.pl_weight = 0 # disable path length regularization
        args.loss_kwargs.style_mixing_prob = 0 # disable style mixing

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        args.batch_size = batch
        args.batch_gpu = batch // gpus

    if gamma is not None:
        assert isinstance(gamma, float)
        if not gamma >= 0:
            raise UserError('--gamma must be non-negative')
        desc += f'-gamma{gamma:g}'
        args.loss_kwargs.r1_gamma = gamma

    # ---------------------------------------------------
    # Discriminator augmentation: aug, p, target, augpipe
    # ---------------------------------------------------

    if aug is None:
        aug = 'ada'
    else:
        assert isinstance(aug, str)
        desc += f'-{aug}'

    if aug == 'ada':
        args.ada_target = 0.6

    elif aug == 'noaug':
        pass

    elif aug == 'fixed':
        if p is None:
            raise UserError(f'--aug={aug} requires specifying --p')

    else:
        raise UserError(f'--aug={aug} not supported')

    if p is not None:
        assert isinstance(p, float)
        if aug != 'fixed':
            raise UserError('--p can only be specified with --aug=fixed')
        if not 0 <= p <= 1:
            raise UserError('--p must be between 0 and 1')
        desc += f'-p{p:g}'
        args.augment_p = p

    if target is not None:
        assert isinstance(target, float)
        if aug != 'ada':
            raise UserError('--target can only be specified with --aug=ada')
        if not 0 <= target <= 1:
            raise UserError('--target must be between 0 and 1')
        desc += f'-target{target:g}'
        args.ada_target = target

    assert augpipe is None or isinstance(augpipe, str)
    if augpipe is None:
        augpipe = 'bgc'
    else:
        if aug == 'noaug':
            raise UserError('--augpipe cannot be specified with --aug=noaug')
        desc += f'-{augpipe}'

    augpipe_specs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }

    assert augpipe in augpipe_specs
    if aug != 'noaug':
        args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[augpipe])

    if nobench is None:
        nobench = False
    assert isinstance(nobench, bool)
    if nobench:
        args.cudnn_benchmark = False

    if allow_tf32 is None:
        allow_tf32 = False
    assert isinstance(allow_tf32, bool)
    if allow_tf32:
        args.allow_tf32 = True

    if workers is not None:
        assert isinstance(workers, int)
        if not workers >= 1:
            raise UserError('--workers must be at least 1')
    
    # Set up categories to train
    if classes is None:
        start = 0
        end = args.training_set_opts.n_unseen
    else:
        l = classes.split('-')
        assert len(l) == 2
        start, end = l
        start = int(start)
        end = int(end)
        desc += f'-{classes}'
    return desc, args, stage_args, (start, end)

def setup_stage_kwargs(args, stage_name, configs): # Set up configs for different stages
    lr = configs['lr']
    kimg = configs['kimg']
    mag = configs['save_magnitude'] if 'save_magnitude' in configs.keys() else [0.1, 0.3, 0.5, 0.7, 0.9, 1]

    if lr is not None:
        args.G_opt_kwargs.lr = lr
        args.D_opt_kwargs.lr = lr
    else:
        args.G_opt_kwargs.lr = 0.0025
        args.D_opt_kwargs.lr = 0.0025

    if kimg is not None:
        args.total_kimg = kimg
    else:
        args.total_kimg = 1

    # Define regularization of seen categories
    sreg_args = dnnlib.EasyDict()
    sreg_args.Greg = True
    sreg_args.Dreg = True

    perc_opts = dnnlib.EasyDict()
    if stage_name == 'LSR':
        if args.k_shot == 1:
            args.ada_kimg = 2
        perc_opts.id_path = path_configs['id_path'] \
            if args.training_set_opts.dataset_name == 'vggfaces' else None
        perc_opts.moco_path = path_configs['moco_path'] \
            if args.training_set_opts.dataset_name in ['flowers', 'animalfaces'] else None
        
    sreg_args.perc_opts = perc_opts
    args.loss_kwargs.sreg_args = sreg_args

    common_args = copy.deepcopy(loss_configs['common'])
    common_args.update(loss_configs[stage_name])
    loss_lambda_dict = dnnlib.EasyDict(**common_args)
    args.loss_kwargs.loss_lambda_dict = loss_lambda_dict

    args.visualization_kwargs.save_magnitudes = mag

    return args

def setup_dataset_kwargs(args, c_idx, idx_list):
    args.training_set_opts.c_idx = c_idx
    args.training_set_opts.idx = idx_list
    return args

def fetch_data(ds, c_idx, idx_list):
    img_list = []
    ws_list = []
    n_unseen = ds.opts.n_unseen_samples
    for idx in idx_list:
        img, ws = ds[c_idx * n_unseen + idx]
        img_list.append(img)
        ws_list.append(ws)
    all_imgs = torch.stack(img_list, dim=0)
    all_ws = torch.stack(ws_list, dim=0)
    return (all_imgs, all_ws)

def run_loops(args, stage_args, ds, c_idx, idx_list):
    # Update dataset args.
    args = setup_dataset_kwargs(args, c_idx, idx_list)
    stage_names = list(stage_args.keys())
    data = fetch_data(ds, c_idx, idx_list)
    with tempfile.TemporaryDirectory() as temp_dir:
        models = [None, None, None, None]
        for i, name in enumerate(stage_names):
            # Set up for each stges
            args = setup_stage_kwargs(args, name, stage_args[name])
            args.models = models
            args.stage_opts = dnnlib.EasyDict()
            args.stage_opts.end = (i == len(stage_names) - 1)
            args.stage_opts.stage_name = name
            if args.global_opts.single_task:
                print()
                print(f'STAGE {name}', '-'*30)
                print()
            stats, models = subprocess_fn(rank=0, data=data, args=args, temp_dir=temp_dir)
        loss_names = []
        for k in stats.as_dict().keys():
            if 'Loss/' in k:
                loss_names.append(k)
        postfix_list = [f"{'_'.join(name.split('/')[1:])}:{stats[name]:<7.4f}" for name in loss_names]
        postfix = ' '.join(postfix_list)
    return postfix

#----------------------------------------------------------------------------

def subprocess_fn(rank, data, args, temp_dir):
    training_stats.reset()
    training_stats._sync_called = False
    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    return training_loop.training_loop(rank=rank, data=data, **args)

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context

# General options.
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--gpus', help='Number of GPUs to use [default: 1]', type=int, metavar='INT')
@click.option('--snap', help='Snapshot interval [default: 1 ticks]', type=int, metavar='INT')
@click.option('--seed', help='Random seed [default: 0]', type=int, metavar='INT')
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)

# Base config.
@click.option('--cfg', help='Base config [default: auto]', type=click.Choice(['auto', 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar']))
@click.option('--gamma', help='Override R1 gamma', type=float)
@click.option('--batch', help='Override batch size', type=int, metavar='INT', default=16)

# Discriminator augmentation.
@click.option('--aug', help='Augmentation mode [default: ada]', type=click.Choice(['noaug', 'ada', 'fixed']))
@click.option('--p', help='Augmentation probability for --aug=fixed', type=float)
@click.option('--target', help='ADA target value for --aug=ada', type=float)
@click.option('--augpipe', help='Augmentation pipeline [default: bgc]', type=click.Choice(['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']))

@click.option('--nobench', help='Disable cuDNN benchmarking', type=bool, metavar='BOOL')
@click.option('--allow-tf32', help='Allow PyTorch to use TF32 internally', type=bool, metavar='BOOL')
@click.option('--workers', help='Override number of DataLoader workers', type=int, metavar='INT')

# Few-shot settings.
@click.option('--dataset_name', help='Name of dataset name [flowers, animalfaces, vggfaces]', required=True, type=click.Choice(['flowers', 'animalfaces', 'vggfaces']))
@click.option('--single_task', help='Perform single batch few-shot generation [default: None] --single_task class_index sample_idx1,sample_idx2', type=(int, str), default=None)
@click.option('--save_ckpt', help='Save ckpts for each tuning.', type=bool, metavar='BOOL', default=False) # Don't set for multi tasks. Takes up disk space to save all the ckpts for each task.
@click.option('--k_shot', help='Number of shots', type=int)
@click.option('--classes', help='Category indexes to optimize [e.g. 0-1]', type=str, default=None)

def main(ctx, outdir, dry_run, **config_kwargs):

    """
    Few-shot Image Generation
    "Where is My Spot? Few-shot Image Generation via Latent Subspace Optimization".

    
    Templates:

    python train_unseen.py \\
        --outdir <output_dir> \\
        --k_shot <k> \\
        --dataset_name <dataset_name> \\
        --classes <start_idx>-<end_idx>

            
    Examples:

    \b
    # Generate images for flowers.

        python train_unseen.py \\
            --outdir ./output/ \\
            --k_shot 1 \\
            --dataset_name flowers

        python train_unseen.py \\
            --outdir ./output/ \\
            --k_shot 3 \\
            --dataset_name flowers

    \b
    # Generate images for animalfaces.

        python train_unseen.py \\
            --outdir ./output/ \\
            --k_shot 1 \\
            --dataset_name animalfaces

        python train_unseen.py \\
            --outdir ./output/ \\
            --k_shot 3 \\
            --dataset_name animalfaces

    \b
    # Generate images for vggfaces.

        python train_unseen.py \\
            --outdir ./output/ \\
            --k_shot 1 \\
            --dataset_name vggfaces

        python train_unseen.py \\
            --outdir ./output/ \\
            --k_shot 3 \\
            --dataset_name vggfaces

    \b
    # Generate images for separate categories (unseen categories [0, 8) and [8, 17) of flowers as examples).

        python train_unseen.py \\
            --outdir ./output/ \\
            --k_shot 1 \\
            --dataset_name flowers \\
            --classes 0-8
        
        Saved to ./output/00000-flowers-1_shot/

        python train_unseen.py \\
            --outdir ./output/ \\
            --k_shot 1 \\
            --dataset_name flowers \\
            --classes 8-17

        Saved to ./output/00001-flowers-1_shot/

        The outputs can be merged using merger.py.

        python merger.py \\
            --path ./output \\
            --idx 0,1

        Merged outputs are stored in ./output/00002-merger_0_1/

    """

    dnnlib.util.Logger(should_flush=True)

    # Deal with conflicts of arguments.
    if config_kwargs['single_task'] is not None:
        class_idx, sample_idxes = config_kwargs['single_task']
        sample_idxes = [int(i) for i in sample_idxes.split(',')]
        config_kwargs['single_task'] = True
        assert config_kwargs['classes'] is None
        assert config_kwargs['k_shot'] == len(sample_idxes)
    else:
        config_kwargs['single_task'] = False
    
    # Setup training options for both stages.
    try:
        run_desc, args, stage_args, (start_cidx, end_cidx) = setup_general_kwargs(**config_kwargs)
    except UserError as err:
        ctx.fail(err)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)

    # Save configs.
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        args_save = copy.deepcopy(args)
        args_save.stage_configs = stage_args
        args_save.loss_configs = loss_configs
        args_save.path_configs = path_configs
        json.dump(args_save, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')

    # Set up image datasets from outer loop.
    ds = AllImagesDataset(args.training_set_opts)

    # Only support single GPU training.
    if args.num_gpus == 1:
        if args.global_opts.single_task:
            args.k_shot = len(sample_idxes)
            run_loops(args, stage_args, ds, class_idx, sample_idxes)
            
        else:
            # Train from category[start_cidx] to category[end_cidx].
            for c_idx in range(start_cidx, end_cidx):
                ibar = tqdm(range(args.training_set_opts.n_unseen_samples))
                ibar.set_description(f'class {c_idx} [{c_idx + 1}/{args.training_set_opts.n_unseen}]')
                for idx in ibar:
                    # Sample k shot tasks.
                    # To alleviate optimization burden, sample n_unseen_samples tasks, with each task containing the adjacent images as inputs.
                    k = args.k_shot
                    postfix = run_loops(args, stage_args, ds, c_idx, \
                        [i % args.training_set_opts.n_unseen_samples for i in range(idx - k // 2, idx + k - k // 2)])
                    ibar.set_postfix({'idx': f'{idx:3d}'})
    
    # Do not support multi-GPU training.
    else:
        print('Only support single GPU training.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
