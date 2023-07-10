import os
import sys
import datetime
import warnings
from pprint import pprint
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.multiprocessing import spawn

from .data.gen_fields import GenerateFieldDataset
from .data import norms
from . import models
from .utils import import_attr, load_model_state_dict

def node_worker(args):
    if 'SLURM_STEP_NUM_NODES' in os.environ:
        args.nodes = int(os.environ['SLURM_STEP_NUM_NODES'])
    elif 'SLURM_JOB_NUM_NODES' in os.environ:
        args.nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    else:
        raise KeyError('missing node counts in slurm env')
    args.threads_per_node = int(os.environ['SLURM_CPUS_ON_NODE'])
    if args.num_threads != -1 :
        if args.threads_per_node > args.num_threads :
            args.threads_per_node = arg.num_threads
    if torch.cuda.is_available():
        args.gpus_per_node = torch.cuda.device_count()
        args.workers_per_node = args.gpus_per_node
        args.world_size = args.nodes * args.gpus_per_node
        args.dist_backend = 'nccl'
    else :
        args.workers_per_node = 1
        args.world_size = args.nodes
        args.dist_backend = 'gloo'
    node = int(os.environ['SLURM_NODEID'])
    spawn(worker, args=(node, args), nprocs=args.workers_per_node)

def worker(local_rank, node, args):

    if torch.cuda.is_available():
        os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        #os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
        device = torch.device('cuda', local_rank)
        torch.backends.cudnn.benchmark = True
        rank = args.gpus_per_node * node + local_rank
    else:  # CPU multithreading
        device = torch.device('cpu')
        rank = local_rank
        torch.set_num_threads(args.threads_per_node)

    dist_file = os.path.join(os.getcwd(), 'dist_addr')
    dist.init_process_group(
        backend=args.dist_backend,
        init_method='file://{}'.format(dist_file),
        world_size=args.world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=14400)
    )
    dist.barrier()
    if rank == 0:
        os.remove(dist_file)

    if args.verbose :
        print('pytorch {}'.format(torch.__version__))
        print()
        pprint(vars(args))
        print()
        sys.stdout.flush()

    generate_dataset = GenerateFieldDataset(
        style_pattern=args.style_pattern,
        pk_pattern=args.pk_pattern,
        out_pattern=args.out_pattern,
        num_mesh_1d=args.num_mesh_1d,
        device = device,
        sphere_mode=args.sphere_mode,
        crop=args.crop,
    )

    generate_loader = DataLoader(
        generate_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=True,
    )

    style_size = generate_dataset.style_size
    in_chan = generate_dataset.in_chan
    out_chan = in_chan

    if args.no_dis :
        d2d_model = None
    else :
        d2d_model = import_attr("nbody.StyledVNet", models)
        d2d_model = d2d_model(style_size, in_chan, out_chan)
        d2d_model.to(device)
        state = torch.load(os.path.dirname(__file__) + "/model_parameters/d2d_params.pt", map_location=device)
        load_model_state_dict(d2d_model, state['model'])
        d2d_model.eval()

    if args.no_vel :
        v2v_model = None
    else :
        v2v_model = import_attr("nbody.StyledVNet", models)
        v2v_model = v2v_model(style_size, in_chan, out_chan)
        v2v_model.to(device)
        state = torch.load(os.path.dirname(__file__) + "/model_parameters/v2v_params.pt", map_location=device)
        load_model_state_dict(v2v_model, state['model'])
        v2v_model.eval()

    generate(generate_loader, d2d_model, v2v_model, rank, args.world_size, device)

def generate(generate_loader, d2d_model, v2v_model, rank, world_size, device) :

    if d2d_model is not None and v2v_model is not None :
        write_chan = (3,) * 2
    else :
        write_chan = (3,)

    with torch.no_grad() :

        remainder = generate_loader.dataset.nout % world_size
        num_mocks = generate_loader.dataset.nout // world_size + 1 if rank < remainder else generate_loader.dataset.nout // world_size
        start_mock = rank * num_mocks if rank < remainder else rank * num_mocks + remainder
        end_mock = start_mock + num_mocks
        if rank == 0 :
            pbar = tqdm(total = num_mocks)
            pbar.set_description(f"Generating linear fields")
        for mock in range(start_mock, end_mock) :
            generate_loader.dataset.generate_linear_field(mock, device)
            if rank == 0 :
                pbar.update(1)

        dis_norm = torch.ones(1, dtype=torch.float64)
        norms.cosmology.dis(dis_norm)
        dis_norm = dis_norm.to(device, non_blocking=True)

        vel_in_norm = torch.ones(1, dtype=torch.float64)
        norms.cosmology.vel(vel_in_norm, dis_std=1)
        vel_in_norm = vel_in_norm.to(device, non_blocking=True)

        vel_norm = torch.ones(1, dtype=torch.float64)
        norms.cosmology.vel(vel_norm)
        vel_norm = vel_norm.to(device, non_blocking=True)

        dist.barrier()

        if rank == 0 :
            pbar = tqdm(total = len(generate_loader))
            pbar.set_description(f"Generating nonlinear fields")

        for i, data in enumerate(generate_loader):

            style, Om, input, out_dir = data['style'], data['Om'], data['input'], data['out_dir']
            style = style.to(device, non_blocking=True)
            input = input.to(device, non_blocking=True)
            input *= dis_norm

            if d2d_model is not None :
                out_dis = d2d_model(input, style)
                out_dis /= dis_norm

            if v2v_model is not None :
                dis2vel = torch.ones(1, dtype=torch.float64)
                norms.cosmology.vel(dis2vel, Om=Om, dis_std=1, undo=True)
                dis2vel = dis2vel.to(device, non_blocking=True)
                input *= dis2vel * vel_in_norm
                out_vel = v2v_model(input, style)
                out_vel /= vel_norm
            
            if d2d_model is not None and v2v_model is not None :
                output = torch.cat((out_dis, out_vel), 1)
                out_paths = [[os.path.join(out_dir[0], 'dis')], [os.path.join(out_dir[0], 'vel')]]
            elif d2d_model is not None :
                output = out_dis
                out_paths = [[os.path.join(out_dir[0], 'dis')]]
            elif v2v_model is not None :
                output = out_vel
                out_paths = [[os.path.join(out_dir[0], 'vel')]]
            generate_loader.dataset.assemble('_out', write_chan, output, out_paths)

            if rank == 0 :
                pbar.update(1)
