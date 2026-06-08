from typing import List, Optional

import numpy as np

from ..data_basic import Dataset


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        # BEGIN YOUR SOLUTION
        import gzip
        import struct
        super().__init__(transforms)
        with gzip.open(image_filename, "rb") as f:
            magic, num_images, rows, cols = struct.unpack(">iiii", f.read(16))
            if magic != 2051:
                raise ValueError(f'无效的图像文件魔数: {magic}')

            # 一次性读取所有像素数据
            buffer = f.read()

            # 将字节数据转换为numpy数组
            self.images = np.frombuffer(buffer, dtype=np.uint8).reshape(
                num_images, rows * cols).astype(np.float32) / 255.0

        with gzip.open(label_filename, "rb") as f:
            magic, num_labels = struct.unpack(">ii", f.read(8))
            if magic != 2049:
                raise ValueError(f'无效的图像文件魔数: {magic}')

            # 一次性读取所有像素数据
            buffer = f.read()

            # 将字节数据转换为numpy数组
            self.labels = np.frombuffer(
                buffer, dtype=np.uint8).reshape(num_labels)
        # END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        # BEGIN YOUR SOLUTION
        if self.transforms:
            reshape_img = np.reshape(self.images[index], (28, 28, 1))
            tform_img = self.apply_transforms(reshape_img)
            return np.reshape(tform_img, (784, 1)), self.labels[index]
        return self.images[index], self.labels[index]
        # END YOUR SOLUTION

    def __len__(self) -> int:
        # BEGIN YOUR SOLUTION
        return self.labels.shape[0]
        # END YOUR SOLUTION
