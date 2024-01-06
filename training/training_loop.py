'''
Main trianing loop
Code adapted from following paper
"Training Generative Adversarial Networks with Limited Data."
See LICENSES/LICENSE_STYLEGAN2_ADA.txt for original license.
'''

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
import legacy
import torchvision.transforms as transforms

from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from training.images_dataset import LSODataset

#----------------------------------------------------------------------------

def unloader(img):
    img = ((img + 1) / 2).clamp(0, 1)
    tf = transforms.Compose([
        transforms.ToPILImage()
    ])
    return tf(img)

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def training_loop(
    data,
    run_dir                 = '.',      # Output directory.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus].
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = None,     # EMA ramp-up coefficient.
    G_reg_interval          = 4,        # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 10,       # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    nimg_per_tick           = 100,      # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    allow_tf32              = False,    # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    mapping_kwargs          = {},       # Argument to construct an extended mapping network.
    visualization_kwargs    = None,     # Set snapshot grid size.
    training_set_opts       = None,     # Training set options.
    global_opts             = None,     # Global settings for all stages.
    stage_opts              = None,     # Specific settings for each stage.
    models                  = None,     # Models from previous stage, if exist.
    k_shot                  = None,     # Number of images in a task.
    grdseed                 = True,     # Enable grid seed for saving outputs.
):
    # Define stage info
    end = stage_opts.end
    stage_name = stage_opts.stage_name
    seen_batch_size = global_opts.seen_batch_size
    seen_batch_gpu = global_opts.seen_batch_gpu

    verbose = global_opts.single_task # if single task, then verbose the training information
    save_ckpt = global_opts.save_ckpt

    seen_reg = loss_kwargs.sreg_args.Greg or loss_kwargs.sreg_args.Dreg
    training_set_opts.idx = np.array(training_set_opts.idx)

    # Define visualization settings
    n_gen = visualization_kwargs.n_gen
    save_magnitudes = visualization_kwargs.save_magnitudes
    grid_size = visualization_kwargs.grid_size
    
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    fake_imgs, ws = data # images and target latent
    # Constuct label for the novel category, need to extend onehot label to n_seen + 1 dimension.
    cs = torch.eye(training_set_opts.n_seen + 1)[[training_set_opts.n_seen for _ in range(k_shot)]]
    training_set = LSODataset(fake_imgs, ws, cs)
    distribution_sampler = misc.DistributionSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=distribution_sampler, batch_size=batch_size//num_gpus))

    phase_ureal_img, _phase_real_ws = data

    # Make output directories.
    output_dir = os.path.join(run_dir, 'few-shot_samples')
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0 and verbose:
        stage_dir = os.path.join(run_dir, stage_name)
        os.makedirs(stage_dir, exist_ok=True)
        ugen_dir = os.path.join(stage_dir, 'fakes')
        os.makedirs(ugen_dir, exist_ok=True)
        anchor_dir = os.path.join(stage_dir, 'anchors')
        os.makedirs(anchor_dir, exist_ok=True)
        print('Loading pretrained Generator...')

    # Fetch trainable modules.
    G, D, anchors, magnitude = models

    if stage_name == 'LAL':
        # Initialize the Generator
        with dnnlib.util.open_url(training_set_opts['stylegan_weights']) as f:
            pkl = legacy.load_network_pkl(f)
            G = pkl['G_ema']
            if rank == 0 and verbose:
                print('Loaded G!')
        G.eval()
        if rank == 0 and verbose:
            print('Extending Generator to n_seen + 1 classes...')
        G.c_dim = training_set_opts.n_seen + 1
        G.mapping.c_dim = training_set_opts.n_seen + 1
        additional_weight = torch.cat([G.mapping.embed.weight, torch.randn([512, 1])], dim = 1)
        G.mapping.embed.weight = torch.nn.Parameter(additional_weight)
        mapping = dnnlib.util.construct_class_by_name(**mapping_kwargs, num_ws=G.mapping.num_ws)
        mapping.eval()
        with torch.no_grad():
            for p_new, p in zip(mapping.parameters(), G.mapping.parameters()):
                p_new.copy_(p)
            for b_new, b in zip(mapping.buffers(), G.mapping.buffers()):
                b_new.copy_(b)
        G.mapping = mapping

        G.to(device)

        # Initialize the trainable anchors.
        anchors = dnnlib.util.construct_class_by_name(class_name='training.networks.Anchor', k_shot=k_shot, device=device)
        anchors.to(device)

        # Initialize the intensity magnitude vector. To restrict the magnitude of subspace, multiply an vector with the random noise vector.
        magnitude = torch.nn.Parameter(torch.ones(1)).detach().to(device)
        magnitude.requires_grad = True

    if stage_name == 'LSR':
        # Initialize the Discriminator.
        if rank == 0 and verbose:
            print('Extending Discriminator to n_seen + 1 classes...')
        with dnnlib.util.open_url(training_set_opts['stylegan_weights']) as f:
            pkl = legacy.load_network_pkl(f)
            D = pkl['D']
            if rank == 0 and verbose:
                print('Loaded D!')
        D.c_dim = training_set_opts.n_seen + 1
        D.mapping.c_dim = training_set_opts.n_seen + 1
        additional_weight = torch.cat([D.mapping.embed.weight, torch.randn([512, 1])], dim = 1)
        D.mapping.embed.weight = torch.nn.Parameter(additional_weight)
        
        D.to(device)

        # Initialize the augmentation pipeline.
        if rank == 0 and verbose:
            print('Setting up augmentation...')
        augment_pipe = None
        ada_stats = None
        if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
            augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
            augment_pipe.p.copy_(torch.as_tensor(augment_p))
            if ada_target is not None:
                ada_stats = training_stats.Collector(regex='Loss/signs/real')

    G_ema = copy.deepcopy(G).eval()

    # Distribute across GPUs.
    if rank == 0 and verbose:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    if stage_name == 'LAL':
        module_list = [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), (None, G_ema)]
    else:
        module_list = [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('D', D), (None, G_ema), ('augment_pipe', augment_pipe)]

    for name, module in module_list:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases.
    if rank == 0 and verbose:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(k_shot=k_shot, stage_name=stage_name, device=device, **ddp_modules, **loss_kwargs)
    phases = []
    if stage_name == 'LAL':
        for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval)]:
            anchors_params = [anchors.z_params]
            mapping_params = list(set(module.mapping.parameters()) - set(module.mapping.embed.parameters()))
            embedding_params = list(module.mapping.embed.parameters())

            fc_params = mapping_params + anchors_params
            if reg_interval is None:
                opt = dnnlib.util.construct_class_by_name(params=[{'params': embedding_params}, \
                    {'params': fc_params, 'lr': opt_kwargs.lr},] + \
                    ([{'params': [magnitude], 'lr': opt_kwargs.lr * 25}] if magnitude is not None else []), **opt_kwargs)

                phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
            else: # Lazy regularization.
                mb_ratio = reg_interval / (reg_interval + 1)
                opt_kwargs = dnnlib.EasyDict(opt_kwargs)
                opt_kwargs.lr = opt_kwargs.lr * mb_ratio
                opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
                opt = dnnlib.util.construct_class_by_name(params=[{'params': embedding_params}, \
                    {'params': fc_params, 'lr': opt_kwargs.lr},] + \
                    ([{'params': [magnitude], 'lr': opt_kwargs.lr * 25}] if magnitude is not None else []), **opt_kwargs)
                phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]

    elif stage_name == 'LSR':
        for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
            anchors_params = [anchors.z_params]
            mapping_params = list(set(module.mapping.parameters()) - set(module.mapping.embed.parameters()))
            embedding_params = list(module.mapping.embed.parameters())
            
            if name == 'D':
                fc_params = mapping_params + list(module.b4.fc.parameters()) + list(module.b4.out.parameters()) + anchors_params
            else:
                fc_params = mapping_params + anchors_params
            # fc_params = mapping_params + anchors_params # have some problems for G...
            if reg_interval is None:
                conv_params = list(set(module.parameters()) - set(fc_params) - set(embedding_params))
                opt = dnnlib.util.construct_class_by_name(params=[{'params': conv_params}, \
                    {'params': fc_params, 'lr': opt_kwargs.lr * 1e-2}, \
                    {'params': embedding_params, 'lr': opt_kwargs.lr * 1e-2},] +  \
                    ([{'params': [magnitude], 'lr': opt_kwargs.lr * 1e2}] if magnitude is not None else []), **opt_kwargs)
                phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
            else: # Lazy regularization.
                mb_ratio = reg_interval / (reg_interval + 1)
                opt_kwargs = dnnlib.EasyDict(opt_kwargs)
                opt_kwargs.lr = opt_kwargs.lr * mb_ratio
                opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
                conv_params = list(set(module.parameters()) - set(fc_params) - set(embedding_params))
                opt = dnnlib.util.construct_class_by_name(params=[{'params': conv_params}, \
                    {'params': fc_params, 'lr': opt_kwargs.lr * 1e-2}, \
                    {'params': embedding_params, 'lr': opt_kwargs.lr * 1e-2},] + \
                    ([{'params': [magnitude], 'lr': opt_kwargs.lr * 1e2}] if magnitude is not None else []), **opt_kwargs)
                phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
                phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_z = None
    grid_c = None
    
    if rank == 0 and verbose:
        print('Exporting real unseen images...')
        img_postfix = f"{training_set_opts.c_idx}_[{'_'.join([str(i) for i in training_set_opts.idx])}]"
        save_image_grid(phase_ureal_img[:k_shot].detach().cpu().numpy(), os.path.join(run_dir, f'real_unseen-{img_postfix}.png'), drange=[-1,1], grid_size=[k_shot, 1])
        n_total = grid_size[0] * grid_size[1]
        grid_z = torch.randn([n_total, G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.eye(training_set_opts.n_seen + 1)[[training_set_opts.n_seen for i in range(n_total)]].to(device).split(batch_gpu)
        fake_imgs = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).detach().numpy()
        save_image_grid(fake_imgs, os.path.join(ugen_dir, f'{stage_name}_fakes_init.png'), drange=[-1,1], grid_size=grid_size)

    # Initialize logs.
    if rank == 0 and verbose:
        print('Initializing logs...')

    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0 and verbose:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(stage_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0 and verbose:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            # Split is only to make it compatible the original multi-gpus version.
            phase_ureal_idx, phase_ureal_img, phase_ureal_w, phase_ureal_c = next(training_set_iterator) # unseen real indexes, unseen real images, unseen real ws, unseen real cs
            phase_opt_a = anchors.get_z(phase_ureal_idx).split(batch_gpu)
            phase_ureal_img = (phase_ureal_img.to(device).to(torch.float32)).split(batch_gpu)
            phase_ureal_w = (phase_ureal_w.to(device).to(torch.float32)).split(batch_gpu)
            phase_ureal_c = (phase_ureal_c.to(device).to(torch.float32)).split(batch_gpu)

            unseen_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            unseen_gen_z = [gen_z.split(batch_gpu) for gen_z in unseen_gen_z.split(batch_size)]
            unseen_gen_c = torch.eye(training_set_opts.n_seen + 1)[[training_set_opts.n_seen for _ in range(len(phases) * batch_size)]].to(device)
            unseen_gen_c = [gen_c.split(batch_gpu) for gen_c in unseen_gen_c.split(batch_size)]

            if seen_reg:
                sgen_z = torch.randn([len(phases) * seen_batch_size, G.z_dim], device=device)
                sgen_z = [gen_z.split(seen_batch_gpu) for gen_z in sgen_z.split(seen_batch_size)]
                idx_list = np.random.randint(0, mapping_kwargs.c_dim - 1, len(phases) * seen_batch_size)
                sgen_c = torch.eye(training_set_opts.n_seen + 1)[idx_list].to(device)
                sgen_c = [gen_c.split(seen_batch_gpu) for gen_c in sgen_c.split(seen_batch_size)]
            else:
                sgen_z = [[None for _ in range(seen_batch_size // seen_batch_gpu)] for _ in range(len(unseen_gen_z))]
                sgen_c = [[None for _ in range(seen_batch_size // seen_batch_gpu)] for _ in range(len(unseen_gen_c))]

        # Execute training phases.
        for phase, phase_ugen_z, phase_ugen_c, phase_sgen_z, phase_sgen_c in zip(phases, unseen_gen_z, unseen_gen_c, sgen_z, sgen_c):
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for _round_idx, (ureal_img, opt_a, ureal_c, ureal_w, ugen_z, ugen_c, sgen_z, sgen_c) in enumerate(zip(phase_ureal_img, \
                phase_opt_a, phase_ureal_c, phase_ureal_w, phase_ugen_z, phase_ugen_c, phase_sgen_z, phase_sgen_c)):
                sync = False
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name, ureal_img=ureal_img, opt_a=opt_a, ureal_c=ureal_c, ureal_w=ureal_w, \
                    ugen_z=ugen_z, ugen_c=ugen_c, sgen_z=sgen_z, sgen_c=sgen_c, m=magnitude, sync=sync, gain=gain)
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            with torch.no_grad():
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_beta))
                for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                    b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if stage_name == 'LSR' and (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + nimg_per_tick):
            continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        # torch.cuda.reset_peak_memory_stats()
        if stage_name == 'LSR':
            fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        
        # Print loss and stats.
        loss_information = []
        for n in stats_collector.names():
            if n.split('/')[0] == 'Loss':
                loss_information += [f"{'_'.join(n.split('/')[1:])} {stats_collector[n]:<5.4f}"]
        if rank == 0 and verbose:
            print(' '.join(fields))
            print(' '.join(loss_information))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0 and verbose:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0 and verbose) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            with torch.no_grad():
                fake_imgs = torch.cat([G_ema.synthesis(G_ema.mapping(z, c, magnitude=1.)).cpu() for z, c in zip(grid_z, grid_c)]).detach().numpy()
                anchor_imgs = G_ema.synthesis(G_ema.mapping(anchors.get_z([i for i in range(k_shot)]), \
                    cs.to(device), magnitude=torch.sigmoid(magnitude))).detach().cpu().numpy()
            save_image_grid(fake_imgs, os.path.join(ugen_dir, f'{stage_name}_fakes-{cur_nimg:06d}.png'), drange=[-1,1], grid_size=grid_size)
            save_image_grid(anchor_imgs, os.path.join(anchor_dir, f'{stage_name}_anchors-{cur_nimg:06d}.png'), drange=[-1,1], grid_size=[k_shot, 1])

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if save_ckpt and (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            # snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            snapshot_data = dict()
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)] if stage_name == 'LSR' else [('G', G), ('G_ema', G_ema)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(stage_dir, f"{stage_name}_network-snapshot-{cur_nimg:06d}.pkl")
            if rank == 0 and verbose:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        if end:
            for save_magnitude in save_magnitudes:
                magnitude_dir = os.path.join(output_dir, f'magnitude-{save_magnitude}')
                if not os.path.exists(magnitude_dir):
                    os.makedirs(magnitude_dir, exist_ok=True)
            G_save = G_ema
            G_save.eval()
            with torch.no_grad():
                if verbose:
                    for i in range(n_gen):
                        z = torch.randn([1, G_save.z_dim], device=device)
                        c = torch.eye(training_set_opts.n_seen + 1)[[training_set_opts.n_seen]].to(device)
                        for save_magnitude in save_magnitudes:
                            img = G_save.synthesis(G_save.mapping(z, c, magnitude=save_magnitude)).detach().cpu()
                            img = unloader(img[0])
                            img.save(os.path.join(output_dir, f'magnitude-{save_magnitude}', '{}_{}.png'.format(training_set_opts.c_idx, str(i).zfill(3))), 'png')
                else:
                    idx_bound = n_gen % training_set_opts.n_unseen_samples
                    if training_set_opts.idx[len(training_set_opts.idx) // 2] < idx_bound:
                        n_samples = n_gen // training_set_opts.n_unseen_samples + 1
                    else:
                        n_samples = n_gen // training_set_opts.n_unseen_samples
                    if grdseed:
                        torch.manual_seed(training_set_opts.c_idx * training_set_opts.n_unseen_samples + \
                            training_set_opts.idx[len(training_set_opts.idx) // 2])
                    z = torch.randn([n_samples, G_save.z_dim], device=device)
                    c = torch.eye(training_set_opts.n_seen + 1)[[training_set_opts.n_seen for i in range(n_samples)]].to(device)
                    for save_magnitude in save_magnitudes:
                        imgs = G_save.synthesis(G_save.mapping(z, c, magnitude=save_magnitude)).detach().cpu()
                        for i, img in enumerate(imgs):
                            img = unloader(img)
                            img.save(os.path.join(output_dir, f'magnitude-{save_magnitude}', '{}_{}.png'.format(training_set_opts.c_idx, \
                                str(i * training_set_opts.n_unseen_samples + training_set_opts.idx[len(training_set_opts.idx) // 2]).zfill(3))), 'png')
        # Done.
        if rank == 0 and verbose:
            print()
            print('Exiting...')

        return stats_collector, [G, D if stage_name == 'LSR' else None, anchors, magnitude] # return for next stages.

#----------------------------------------------------------------------------
