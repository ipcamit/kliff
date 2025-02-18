from pathlib import Path
from typing import Callable, Optional, Union

import torch
from torch.utils.data import Dataset

import pickle

# loading the older descriptor module sometimes creates clashes
# from kliff.legacy.descriptors.descriptor import load_fingerprints

def load_fingerprints(path: Union[Path, str]):
    """
    Read preprocessed fingerprints from file.

    This is the reverse operation of Descriptor._dump_fingerprints.

    Args:
        path: Path to the pickled data file.

    Returns:
        Fingerprints
    """
    data = []
    with open(path, "rb") as f:
        try:
            while True:
                x = pickle.load(f)
                data.append(x)
        except EOFError:
            pass

    return data


class FingerprintsDataset(Dataset):
    """
    Atomic environment fingerprints dataset used by torch models.

    Args:
        filename: to the fingerprints file.
        transform: transform to be applied on a sample.
    """

    def __init__(self, filename: Path, transform: Optional[Callable] = None):
        self.fp = load_fingerprints(filename)
        self.transform = transform

    def __len__(self):
        return len(self.fp)

    def __getitem__(self, index):
        sample = self.fp[index]
        if self.transform:
            sample = self.transform(sample)
        return sample


def fingerprints_collate_fn(batch):
    """
    Convert a batch of samples into tensor.

    Unlike the default collate_fn(), which stack samples in the batch (requiring each
    sample having the same dimension), this function does not do the stack.

    Args:
        batch: A batch of samples.

    Returns:
        A list of tensor.
    """
    tensor_batch = []
    for i, sample in enumerate(batch):
        tensor_sample = {}
        for key, value in sample.items():
            if type(value).__module__ == "numpy":
                value = torch.from_numpy(value)
            tensor_sample[key] = value
        tensor_batch.append(tensor_sample)

    return tensor_batch
