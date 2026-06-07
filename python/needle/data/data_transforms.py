import numpy as np


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C NDArray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        # BEGIN YOUR SOLUTION
        if flip_img:
            return np.flip(img, axis=1)
        else:
            return img
        # END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NDArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding + 1, size=2)
        # BEGIN YOUR SOLUTION
        shape0, shape1, _ = img.shape
        padding = self.padding
        x, y = padding + shift_x, padding + shift_y
        img = np.pad(img, pad_width=((padding, padding), (padding, padding), (0, 0)))
        return img[x:x + shape0, y:y + shape1, :]
        # END YOUR SOLUTION
