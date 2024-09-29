# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main training loop."""

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
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main
from camera_utils import LookAtPoseSampler
from dataset_tool import sample_frames

#----------------------------------------------------------------------------

def get_id_param():
    id_param = np.load('factors_id_2548_50.npy')
    return id_param

def setup_snapshot_image_grid(training_set, random_seed=0, bias=30):
    rnd = np.random.RandomState(random_seed)
    # gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    # gh = np.clip(4320 // training_set.image_shape[1], 4, 32)
    gw = 1
    gh = 1

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        # label_groups = dict() # label => [idx, ...]
        # for idx in range(len(training_set)):
        #     label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
        #     if label not in label_groups:
        #         label_groups[label] = []
        #     label_groups[label].append(idx)
        #
        # # Reorder.
        # label_order = list(label_groups.keys())
        # rnd.shuffle(label_order)
        # for label in label_order:
        #     rnd.shuffle(label_groups[label])

        # Organize into grid.
        # grid_indices = [(i * (53 * 72) + bias) for i in range(2)]#multi people
        # grid_indices = [i * 3744 + 3702 for i in range(50)]
        # grid_indices = [(i * 14 + bias) for i in range(52)]# single person 14 frames
        # grid_indices = [(i * 72) + bias for i in range(50)]#single people 72 frames
        grid_indices = [bias]
        # for y in range(gh):
        #     # label = label_order[y % len(label_order)]
        #     label = label_order[30]#front face
        #     indices = label_groups[label]
        #     grid_indices += [indices[x % len(indices)] for x in range(gw)]
        #     label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels, verts, ids = zip(*[training_set[i] for i in grid_indices])
    touding_label = np.array([1.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, -1.0, 22.5,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 1.0,
                  28.0, 0.0, 0.5,
                  0.0, 28.0, 0.5,
                  0.0, 0.0, 1.0], dtype=np.float32)
    beihou_label = np.array([-1.0, 0.0, 0.0, 0.0,
                  0.0, -1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, -22.5,
                  0.0, 0.0, 0.0, 1.0,
                  28.0, 0.0, 0.5,
                  0.0, 28.0, 0.5,
                  0.0, 0.0, 1.0], dtype=np.float32)
    test_label = beihou_label

    # labels = tuple([test_label] * 4)
    if len(images[0].shape) == 4: # select the first frame for every video pair
        images = [image[0] for image in images]
        labels = [label[0] for label in labels]
        verts = [vert[0] for vert in verts]
    return (gw, gh), np.stack(images), np.stack(labels), np.stack(verts), np.stack(ids)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = False                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
        
    if G_kwargs.rendering_kwargs.get('gen_exp_cond'):
        g_c_dim = 25
        d_c_dim = training_set.label_dim if not 'DualLabel' in D_kwargs.class_name else 25
        c_dim = training_set.label_dim
    else:
        g_c_dim = d_c_dim = training_set.label_dim
    common_kwargs = dict(img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(c_dim=g_c_dim, **G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G.register_buffer('dataset_label_std', torch.tensor(training_set.get_label_std()).to(device))
    if 'Video' in training_set_kwargs['class_name']:
        common_kwargs['img_channels'] *= training_set_kwargs['sampling_dict']['num_frames_per_video']
        d_c_dim *= 2
    if 'DualLabel' in D_kwargs.class_name:
        D_kwargs['c2_dim'] = 52 # expression label dim
    D = dnnlib.util.construct_class_by_name(c_dim=d_c_dim, **D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

        # debug: copy backbone parameters to texture backbone
        try:
            misc.copy_params_and_buffers(resume_data['G'].texture_backbone, G.texture_backbone, require_all=False)
            misc.copy_params_and_buffers(resume_data['G_ema'].texture_backbone, G_ema.texture_backbone, require_all=False)
        except:
            misc.copy_params_and_buffers(resume_data['G'].backbone, G.texture_backbone, require_all=False)
            misc.copy_params_and_buffers(resume_data['G_ema'].backbone, G_ema.texture_backbone, require_all=False)
        if getattr(G, 'mouth_backbone'):
            try:
                misc.copy_params_and_buffers(resume_data['G'].mouth_backbone, G.mouth_backbone, require_all=False)
                misc.copy_params_and_buffers(resume_data['G_ema'].mouth_backbone, G_ema.mouth_backbone, require_all=False)
            except:
                misc.copy_params_and_buffers(resume_data['G'].backbone, G.mouth_backbone, require_all=False)
                misc.copy_params_and_buffers(resume_data['G_ema'].backbone, G_ema.mouth_backbone, require_all=False)
        if getattr(G, 'neural_blending'):
            try:
                misc.copy_params_and_buffers(resume_data['G'].neural_blending, G.neural_blending, require_all=False)
                misc.copy_params_and_buffers(resume_data['G_ema'].neural_blending, G_ema.neural_blending, require_all=False)
            except:
                misc.copy_params_and_buffers(resume_data['G'].backbone, G.neural_blending, require_all=False)
                misc.copy_params_and_buffers(resume_data['G_ema'].backbone, G_ema.neural_blending, require_all=False)
    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        z_noise = torch.empty([batch_gpu, G.z_noise_dim], device=device)
        c_g = torch.empty([batch_gpu, G.c_dim], device=device)
        if "DualLabel" in D_kwargs.class_name:
            c_d = torch.empty([batch_gpu, D.c_dim + D.c2_dim], device=device)
        else:
            c_d = torch.empty([batch_gpu, D.c_dim], device=device)
        v = torch.empty([batch_gpu, 2548, 3], device=device)
        if training_set.load_lms:
            # load landmarks
            lms = np.loadtxt('data/mh/template.txt')#TODO
            lms = torch.from_numpy(lms).to(device).float().unsqueeze(0)
            v = torch.cat((v, lms.repeat(v.shape[0], 1, 1)), 1)
        img = misc.print_module_summary(G, [z, z_noise, c_g, v])
        if 'Video' in training_set_kwargs['class_name']:
            img = {k: torch.cat([v, v], 1) for k, v in img.items()}
        misc.print_module_summary(D, [img, c_d])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    grid_v = None
    # raw_id_list = np.eye(G.z_dim)
    raw_id_list = get_id_param()
    if rank == 0:

        print('Exporting sample images...')
        # print('Exporting sample images...')
        '''for training'''
        grid_size, images, labels, verts, ids = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        grid_z_noise = torch.randn([labels.shape[0], G.z_noise_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        grid_z = torch.from_numpy(raw_id_list[ids]).to(device).split(batch_gpu)
        grid_v = torch.from_numpy(verts).to(device).split(batch_gpu)
        '''for test'''
        # grid_vs = []
        # grid_cs = []
        # grid_zs = []
        # for bias in range(72):
        #     grid_size, images, labels, verts, ids = setup_snapshot_image_grid(training_set=training_set, bias=bias)
        #     # save_image_grid(images, os.path.join(run_dir, 'reals_' + str(bias) + '.png'), drange=[0,255], grid_size=grid_size)
        #     # grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        #     grid_z = torch.from_numpy(raw_id_list[ids]).to(device).split(batch_gpu)
        #     grid_zs.append(grid_z)
        #     grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        #     grid_cs.append(grid_c)
        #     grid_v = torch.from_numpy(verts).to(device).split(batch_gpu)
        #     grid_vs.append(grid_v)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 100#TODO
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:
        t_start = time.time()
        # Fetch training data.
        '''
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c, phase_real_v, phase_real_id = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            t_noise_start = time.time()
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_noise_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            t_noise_end = time.time()

            gen_cond_sample_idx = [np.random.randint(len(training_set)) for _ in range(len(phases) * batch_size)]
            if 'Video' in training_set_kwargs['class_name']:
                all_gen_l = [min(training_set.get_video_len(i), training_set.max_num_frames) for i in gen_cond_sample_idx]
                gen_cond_sample_frame_idx = [sample_frames(training_set.sampling_dict, total_video_len=l) for l in all_gen_l]
                all_gen_c = [training_set.get_label(i, frames_idx) for i, frames_idx in zip(gen_cond_sample_idx, gen_cond_sample_frame_idx)]
                all_gen_v = [training_set.get_vert(i, frames_idx) for i, frames_idx in zip(gen_cond_sample_idx, gen_cond_sample_frame_idx)]
            else:
                all_gen_c = [training_set.get_label(i) for i in gen_cond_sample_idx]
                all_gen_v = [training_set.get_vert(i) for i in gen_cond_sample_idx] # TODO: if fix the index of labels and vertices
            choose_img = [training_set.get_img(i) for i in gen_cond_sample_idx]
            all_choose_img = torch.Tensor(np.stack(choose_img)).pin_memory().to(device)
            all_choose_img = [(phase_choose_img.to(torch.float32) / 127.5 - 1).split(batch_gpu) for phase_choose_img in
                              all_choose_img.split(batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]
            all_gen_id = [training_set.get_id(i) for i in gen_cond_sample_idx]
            all_gen_id = torch.from_numpy(raw_id_list[np.stack(all_gen_id)]).pin_memory().to(device)
            all_gen_id = [phase_gen_id.split(batch_gpu) for phase_gen_id in all_gen_id.split(batch_size)]
            all_gen_v = torch.from_numpy(np.stack(all_gen_v)).pin_memory().to(device)
            all_gen_v = [phase_gen_v.split(batch_gpu) for phase_gen_v in all_gen_v.split(batch_size)]
        # Execute training phases.
        t = []
        t0 = time.time()
        # print(t_start - t0)
        for phase, phase_gen_id, phase_gen_z, phase_gen_c, phase_gen_v, phase_choose_img in zip(phases, all_gen_id, all_gen_z, all_gen_c, all_gen_v, all_choose_img):#TODO
            t1 = time.time()
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for real_imgA, real_imgB, real_c, gen_id, gen_z_noise, gen_c, gen_v in zip(phase_real_img, phase_choose_img, phase_real_c, phase_gen_id, phase_gen_z, phase_gen_c, phase_gen_v):
                loss.accumulate_gradients(phase=phase.name, real_imgA=real_imgA, real_imgB=real_imgB, real_c=real_c, gen_id=gen_id, gen_z_noise=gen_z_noise, gen_c=gen_c, gen_v=gen_v, gain=phase.interval, cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)
            # t2 = time.time()
            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # debug: clamp values of orthogonal parameters
            for name, param in phase.module.named_parameters():
                if name == 'orth_scale':
                    param.data.clamp_(8.90, 9.10)
                elif name == 'orth_shift':
                    param.data.clamp_(-0.04, 0.01)

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))
            t2 = time.time()
            t.append(t2 - t1)
            # print()
        # Update G_ema.
        t3 = time.time()
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 100
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)
            G_ema.neural_rendering_resolution = G.neural_rendering_resolution
            G_ema.rendering_kwargs = G.rendering_kwargs.copy()

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 100)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 100)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 100):
            continue
        t4 = time.time()
        # Print status line, accumulating the same information in training_stats.
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
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        fields += [f"phase_time {t}"]#TODO
        fields += [f"batch_time {t4 - t_start}"]  # TODO
        fields += [f"data_time {t0 - t_start}"]
        fields += [f"noise_time {t_noise_start - t_noise_end}"]
        fields += [f"else_time {t4 - t3}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')
        '''
        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (cur_tick % image_snapshot_ticks == 0):
            '''for training'''
            out = [G_ema(z=z, z_noise=z_noise, c=c, v=v, noise_mode='const') for z, z_noise, c, v in zip(grid_z, grid_z_noise, grid_c, grid_v)]
            images = torch.cat([o['image'].cpu() for o in out]).numpy()
            images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
            images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}.png'), drange=[-1, 1],
                            grid_size=grid_size)
            save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_raw.png'), drange=[-1, 1],
                            grid_size=grid_size)
            save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_depth.png'),
                            drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)
            # # '''for test'''
            # for i in range(72):
            #     grid_z = grid_zs[i]
            #     grid_c = grid_cs[i]
            #     grid_v = grid_vs[i]
            #     out = [G_ema(z=z, c=c, v=v, noise_mode='const') for z, c, v in zip(grid_z, grid_c, grid_v)]
            #     images = torch.cat([o['image'].cpu() for o in out]).numpy()
            #     images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
            #     images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
            #     save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_' + str(i) + '.png'), drange=[-1,1], grid_size=grid_size)
                # save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw_' + str(i) + '.png'), drange=[-1,1], grid_size=grid_size)
                # save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth_' + str(i) + '.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)


            #--------------------
            # # Log forward-conditioned images

            # forward_cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
            # intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
            # forward_label = torch.cat([forward_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            # grid_ws = [G_ema.mapping(z, forward_label.expand(z.shape[0], -1)) for z, c in zip(grid_z, grid_c)]
            # out = [G_ema.synthesis(ws, c=c, noise_mode='const') for ws, c in zip(grid_ws, grid_c)]

            # images = torch.cat([o['image'].cpu() for o in out]).numpy()
            # images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
            # images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
            # save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_f.png'), drange=[-1,1], grid_size=grid_size)
            # save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw_f.png'), drange=[-1,1], grid_size=grid_size)
            # save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth_f.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)

            #--------------------
            # # Log Cross sections

            # grid_ws = [G_ema.mapping(z, c.expand(z.shape[0], -1)) for z, c in zip(grid_z, grid_c)]
            # out = [sample_cross_section(G_ema, ws, w=G.rendering_kwargs['box_warp']) for ws, c in zip(grid_ws, grid_c)]
            # crossections = torch.cat([o.cpu() for o in out]).numpy()
            # save_image_grid(crossections, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_crossection.png'), drange=[-50,100], grid_size=grid_size)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        training_set_kwargs_tmp = training_set_kwargs.copy()
        training_set_kwargs_tmp['load_obj'] = False
        training_set_kwargs_tmp['load_lms'] = False
        if (snapshot_data is not None) and (len(metrics) > 0) and (cur_tick >= 0):
            if rank == 0:
                print(run_dir)
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                    dataset_kwargs=training_set_kwargs_tmp, num_gpus=num_gpus, rank=rank, device=device, cond_vert=True)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
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
            progress_fn(cur_nimg // 100, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
