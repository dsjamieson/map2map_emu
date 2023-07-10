from .fields import *

class RunFieldDataset(FieldDataset):
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
    def __init__(self, style_pattern, in_pattern, out_pattern, 
                 callback_at=None, crop=None, crop_start=None, crop_stop=None, crop_step=None,
                 in_pad=48, tgt_pad=0, scale_factor=1, **kwargs) :

        self.style_files = sorted(glob(style_pattern))
        self.in_files = sorted(glob(in_pattern))
        self.out_dirs = sorted(glob(out_pattern))
        self.nout = len(self.out_dirs)
        if self.nout == 0 :
            raise FileNotFoundError('Dirs not found for {}'.format(out_pattern))
        self.is_read = np.full(self.nout, False)

        if len(self.style_files) != 1 and len(self.style_files) != self.nout :
            raise ValueError('number of style files and output dirs do not match')
        if len(self.style_files) == 1 :
            self.style_files = [self.style_files[0]] * self.nout            
        if len(self.in_files) != self.nout :
            raise ValueError('number of input files and output dirs do not match')

        self.style_col = [0]
        self.style_size = np.loadtxt(self.style_files[0])[self.style_col].shape[0]
        self.in_chan = 3
        self.size = np.load(self.in_files[0], mmap_mode='r').shape[1:]
        self.size = np.asarray(self.size)
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
            
    def __len__(self):
        return self.nsample

    def __getitem__(self, idx):
        
        mock, crop_num = divmod(idx, self.ncrop)

        # use memmap after reading a file once
        if not self.is_read[mock] :
            mmap_mode = None
            self.is_read[mock] = True
        else:
            mmap_mode = 'r'
            
        dis_in_filename = self.in_files[mock]
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

        return {
            'style': style,
            'input': in_fields,
            'out_dir': self.out_dirs[mock],
            'Om' : Om[0]
        }
