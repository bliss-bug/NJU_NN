import gzip
import math
import struct
from typing import Optional, List
import numpy as np
from autograd import Tensor


class Dataset:
    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def apply_transforms(self, x):
        if self.transforms is not None:
            for tform in self.transforms:
                x = tform(x)
        return x



class DataLoader:
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.batch_size = batch_size
        if not shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))
        else:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)), 
                                           range(self.batch_size, len(self.dataset), self.batch_size))

    def __iter__(self):
        self.idx = -1
        return self

    def __next__(self):
        self.idx += 1
        if self.idx >= len(self.ordering):
            raise StopIteration
        samples = [self.dataset[i] for i in self.ordering[self.idx]]
        return [Tensor(np.stack([samples[i][j] for i in range(len(samples))])) for j in range(len(samples[0]))]
    
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
    


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.transforms = transforms
        with gzip.open(image_filename, "rb") as img_file:
            magic_num, img_num, row, col = struct.unpack(">4i", img_file.read(16))
            assert(magic_num == 2051)
            tot_pixels = row * col
            imgs = [np.array(struct.unpack(f"{tot_pixels}B",
                                           img_file.read(tot_pixels)),
                                           dtype=np.float32)
                    for _ in range(img_num)]
            X = np.vstack(imgs)
            X -= np.min(X)
            X /= np.max(X)
            self.X = X

        with gzip.open(label_filename, "rb") as label_file:
            magic_num, label_num = struct.unpack(">2i", label_file.read(8))
            assert(magic_num == 2049)
            self.y = np.array(struct.unpack(f"{label_num}B", label_file.read()), dtype=np.uint8)

    def __getitem__(self, index):
        imgs = self.X[index]
        labels = self.y[index]
        if len(imgs.shape) > 1:
            imgs = np.vstack([self.apply_transforms(img) for img in imgs])
        else:
            imgs = self.apply_transforms(imgs)
        return (imgs, labels)

    def __len__(self):
        return self.X.shape[0]



class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self):
        return self.arrays[0].shape[0]

    def __getitem__(self, idx):
        return tuple([a[idx] for a in self.arrays])