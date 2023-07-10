from .fields import *

def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    '''
    One-dimensional linear interpolation for monotonically increasing sample points assuming
    x is at least a 2D grid.
    '''
    
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])  # Slope
    b = fp[:-1] - (m * xp[:-1])
    
    indices = torch.zeros(x.shape, dtype = torch.int32)
    for i in range(indices.shape[0]) :
        indices[i] = torch.sum(torch.ge(x[i][:,:, None], xp[None, None, :]), -1) - 1

    return m[indices] * x + b[indices]

def pk_to_dis(k: torch.Tensor, p: torch.Tensor, scalar_potential: torch.Tensor) -> torch.Tensor:
    
    device = k.device
    
    num_mesh_1d = scalar_potential.shape[0]
    box_length = 1.e3 * num_mesh_1d / 512
    box_volume = box_length ** 3
    bin_volume = (box_length / num_mesh_1d) ** 3
    fundamental_mode = 2. * torch.pi / box_length
    num_modes_last_d = (num_mesh_1d // 2) + 1

    log_kk = torch.log(k)
    log_sig = 0.5 * torch.log(p * box_length ** 3)
    
    wave_numbers = torch.fft.fftfreq(num_mesh_1d, d = 1. / fundamental_mode / num_mesh_1d).to(device)
    k_squared_grid = wave_numbers.unsqueeze(1).unsqueeze(1).pow(2) + wave_numbers.unsqueeze(1).unsqueeze(0).pow(2) \
        + wave_numbers[:num_modes_last_d].unsqueeze(0).unsqueeze(0).pow(2)
    k_squared_grid[0,0,0].fill_(1)
    log_sig_grid = 0.5 * torch.log(k_squared_grid)
    sig_grid = interpolate(log_sig_grid, log_kk, log_sig)
    sig_grid = torch.exp(sig_grid)
    sig_grid[0,0,0].fill_(0)
    
    scalar_potential = torch.fft.rfftn(scalar_potential) / num_mesh_1d ** 1.5
    
    scalar_potential = scalar_potential * sig_grid
    scalar_potential = scalar_potential / (-1j * k_squared_grid * bin_volume)
    scalar_potential[0,0,0].fill_(0.)
    
    dis_in = torch.zeros((3, num_mesh_1d, num_mesh_1d, num_mesh_1d))
    dis_in[0,:,:,:] = torch.fft.irfftn(wave_numbers[:,None,None] * scalar_potential)
    dis_in[1,:,:,:] = torch.fft.irfftn(wave_numbers[None,:,None] * scalar_potential)
    dis_in[2,:,:,:] = torch.fft.irfftn(wave_numbers[None,None,:num_modes_last_d] * scalar_potential)
    
    return dis_in

class GenerateFieldDataset(FieldDataset):
    """Dataset of lists of fields.

    `in_pattern` is a list of glob pattern for the input field files.
    For example, `in_pattern=['/train/field1_*.npy', '/train/field2_*.npy']`.
    Each pattern in the list is a new field.
    Input and target fields are matched by sorting the globbed files.

    `in_norms` is a list of of functions to normalize the input fields.

    NOTE that vector fields are assumed if numbers of channels and dimensions are equal.

    Input and target fields can be cropped, to return multiple slices of size
    `crop` from each field.
    The crop anchors are controlled by `crop_start`, `crop_stop`, and `crop_step`.
    Input (but not target) fields can be padded beyond the crop size assuming
    periodic boundary condition.

    Setting integer `scale_factor` greater than 1 will crop target bigger than
    the input for super-resolution, in which case `crop` and `pad` are sizes of
    the input resolution.
    """
    def __init__(self, style_pattern, pk_pattern, out_pattern, num_mesh_1d, device, sphere_mode = False,
                 callback_at=None, crop=None, crop_start=None, crop_stop=None, crop_step=None,
                 in_pad=0, tgt_pad=0, scale_factor=1, **kwargs):

        #self.printed = True

        self.style_files = sorted(glob(style_pattern))
        self.pk_files = sorted(glob(pk_pattern))
        self.out_dirs = sorted(glob(out_pattern))
        self.nout = len(self.out_dirs)
        if self.nout == 0:
            raise FileNotFoundError('Dirs not found for {}'.format(out_pattern))
        self.is_generated = np.full(self.nout, False)
        self.num_mesh_1d = num_mesh_1d

        if len(self.style_files) != 1 and len(self.style_files) != self.nout :
            raise ValueError('number of style files and output dirs do not match')
        if len(self.style_files) == 1 :
            self.style_files = [self.style_files[0]] * self.nout            
        if len(self.pk_files) != 1 and len(self.pk_files) != self.nout :
            raise ValueError('number of power spectrum files and output dirs do not match')
        if len(self.pk_files) == 1 :
            self.pk_files = [self.pk_files[0]] * self.nout

        self.style_col = [0]
        self.style_size = np.loadtxt(self.style_files[0])[self.style_col].shape[0]
        self.in_chan = 3
        self.size = np.array([num_mesh_1d] * 3)
        self.ndim = len(self.size)

        def format_pad(pad, ndim):
            if isinstance(pad, int):
                pad = np.broadcast_to(pad, ndim * 2)
            elif isinstance(pad, tuple) and len(pad) == ndim:
                pad = np.repeat(pad, 2)
            elif isinstance(pad, tuple) and len(pad) == ndim * 2:
                pad = np.array(pad)
            else:
                raise ValueError('pad and ndim mismatch')
            return pad.reshape(ndim, 2)
        self.in_pad = format_pad(in_pad, self.ndim)
        self.tgt_pad = format_pad(tgt_pad, self.ndim)

        self.callback_at = callback_at

        if crop is None:
            self.crop = self.size
        else:
            self.crop = np.broadcast_to(crop, (self.ndim,))

        if crop_start is None:
            crop_start = np.zeros_like(self.size)
        else:
            crop_start = np.broadcast_to(crop_start, (self.ndim,))

        if crop_stop is None:
            crop_stop = self.size
        else:
            crop_stop = np.broadcast_to(crop_stop, (self.ndim,))

        if crop_step is None:
            crop_step = self.crop
        else:
            crop_step = np.broadcast_to(crop_step, (self.ndim,))
        self.crop_step = crop_step

        self.anchors = np.stack(np.mgrid[tuple(slice(crop_start[d], crop_stop[d], crop_step[d]) for d in range(self.ndim))],
                                axis=-1).reshape(-1, self.ndim)
        self.ncrop = len(self.anchors)

        def format_pad(pad, ndim):
            if isinstance(pad, int):
                pad = np.broadcast_to(pad, ndim * 2)
            elif isinstance(pad, tuple) and len(pad) == ndim:
                pad = np.repeat(pad, 2)
            elif isinstance(pad, tuple) and len(pad) == ndim * 2:
                pad = np.array(pad)
            else:
                raise ValueError('pad and ndim mismatch')
            return pad.reshape(ndim, 2)
        self.in_pad = format_pad(in_pad, self.ndim)

        self.scale_factor = scale_factor

        self.nsample = self.nout * self.ncrop

        self.kwargs = kwargs

        self.assembly_line = {}
     
    def generate_linear_field(self, mock, device) :
        ps_k, ps_p = torch.from_numpy(np.loadtxt(self.pk_files[mock]).T).to(device)
        seed = torch.randint(2 ** 32 - 1, (1,)).item()
        random_generator = torch.Generator()
        random_generator.manual_seed(seed)
        scalar_potential = torch.randn((self.num_mesh_1d,) * 3, generator=random_generator).to(device)
        scalar_potential[self.num_mesh_1d // 2] = 0
        scalar_potential[:,self.num_mesh_1d // 2] = 0
        scalar_potential[:,:,self.num_mesh_1d // 2] = 0
        dis_in = pk_to_dis(ps_k, ps_p, scalar_potential)
        del(ps_k)
        del(ps_p)
        del(scalar_potential)
        seed_filename = self.out_dirs[mock] + "/seed.npy"
        np.save(seed_filename, [seed])
        dis_in_filename = self.out_dirs[mock] + "/dis_in.npy"
        np.save(dis_in_filename, np.float32(dis_in))
            
    def __len__(self):
        return self.nsample

    def __getitem__(self, idx):
        
        mock, crop_num = divmod(idx, self.ncrop)

        # use memmap after reading a file once
        if not self.is_generated[mock] :
            mmap_mode = None
            self.is_generated[mock] = True
        else:
            mmap_mode = 'r'
            
        dis_in_filename = self.out_dirs[mock] + "/dis_in.npy"
        in_fields = [np.load(dis_in_filename, mmap_mode=mmap_mode)]
        Om = np.loadtxt(self.style_files[mock])[self.style_col]

        anchor = self.anchors[crop_num]

        # crop and pad are for the shapes after perm()
        # so before that they themselves need perm() in the opposite ways
        argsort_perm_axes = slice(None)
        crop(in_fields, anchor, self.crop[argsort_perm_axes], self.in_pad[argsort_perm_axes])

        style = torch.tensor(Om).to(torch.float32)
        Om = torch.from_numpy(Om).to(torch.float32)
        in_fields = [torch.from_numpy(f).to(torch.float32) for f in in_fields]

        # HACK
        style -= torch.tensor([0.3])
        style *= torch.tensor([5.0])
        in_fields = torch.cat(in_fields, dim=0)


        #if self.printed :
        #    print(in_fields.shape, crop_num, anchor, in_fields[0,:2,:2,:2])
        #    self.printed = False

        return {
            'style': style,
            'input': in_fields,
            'out_dir': self.out_dirs[mock],
            'Om' : Om[0]
        }
