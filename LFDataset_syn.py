import torch
from torch.utils.data import Dataset
import h5py
import scipy.io as scio
import numpy as np
from Functions import ExtractPatch_d


def _canonical_lf(array, ang_res=5):
    array = np.asarray(array)
    if array.ndim != 6:
        raise ValueError(f"Expected a 6-D light-field array, got {array.shape}")

    # Classic scipy.io.loadmat layout: [u, v, H, W, C, sample].
    if array.shape[0] == ang_res and array.shape[1] == ang_res and array.shape[4] == 3:
        return array.transpose(5, 0, 1, 4, 2, 3)

    # MATLAB v7.3 through h5py normally appears as [sample, C, W, H, v, u].
    if array.shape[-1] == ang_res and array.shape[-2] == ang_res and array.shape[1] == 3:
        return array.transpose(0, 5, 4, 1, 3, 2)

    raise ValueError(
        "Unsupported light-field layout {}. Expected [u,v,H,W,3,N] or "
        "the v7.3/HDF5-reversed [N,3,W,H,v,u].".format(array.shape)
    )


def _load_lf_mat(path):
    try:
        data = scio.loadmat(path)
        lf = data['lf']
        degraded_key = 'lf_degradation' if 'lf_degradation' in data else 'lf_degeneration'
        lowlf = data[degraded_key]
    except (NotImplementedError, ValueError, OSError):
        with h5py.File(path, 'r') as data:
            if 'lf' not in data:
                raise KeyError("MAT file does not contain the 'lf' dataset")
            degraded_key = 'lf_degradation' if 'lf_degradation' in data else 'lf_degeneration'
            if degraded_key not in data:
                raise KeyError("MAT file does not contain 'lf_degradation' or 'lf_degeneration'")
            lf = np.asarray(data['lf'])
            lowlf = np.asarray(data[degraded_key])
    return _canonical_lf(lf), _canonical_lf(lowlf)

class LFDataset(Dataset):
    """Light Field dataset."""
    def __init__(self, opt):
        super(LFDataset, self).__init__()     
        self.lfSet, self.lowlfSet = _load_lf_mat(opt.dataPath) #[ind, u, v, c, x, y]
        # self.lowlfSet = dataSet['lowlf_20'].transpose(5,0,1,4,2,3) #[ind, u, v, c, x, y]
        self.patchSize=opt.patchSize
        self.augment = opt.augment


    def __getitem__(self, idx):
        
        lf=self.lfSet[idx]
        lf = np.array(lf, dtype="float32") / 255.0
        lowlf = self.lowlfSet[idx]
        lowlf = np.array(lowlf, dtype="float32") / 255.0
        H = self.lfSet.shape[4]
        W = self.lfSet.shape[5]

        lfPatch, lowlfPatch =ExtractPatch_d(lf, lowlf, H, W, self.patchSize) #[u v c x y]
        if self.augment:
            if np.random.rand(1) > 0.5:
                lfPatch = np.flip(np.flip(lfPatch, 0), 3)
                lowlfPatch = np.flip(np.flip(lowlfPatch, 0), 3)
            if np.random.rand(1) > 0.5:
                lfPatch = np.flip(np.flip(lfPatch, 1), 4)
                lowlfPatch = np.flip(np.flip(lowlfPatch, 1), 4) 
            # rotate
            r_ang = np.random.randint(1, 5)
            lfPatch = np.rot90(lfPatch, r_ang, (3, 4))
            lfPatch = np.rot90(lfPatch, r_ang, (0, 1))
            lowlfPatch = np.rot90(lowlfPatch, r_ang, (3, 4))
            lowlfPatch = np.rot90(lowlfPatch, r_ang, (0, 1))
            
        lfPatch= torch.from_numpy(lfPatch)
        lowlfPatch= torch.from_numpy(lowlfPatch)
        sample = {'lf':lfPatch,'lowlf':lowlfPatch}
        return sample
        
    def __len__(self):
        return self.lfSet.shape[0]
