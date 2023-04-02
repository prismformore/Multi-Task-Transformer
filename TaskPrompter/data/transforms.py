# same transform as ATRC

import numpy as np
import random
import cv2
import torch


class RandomScaling:
    """Random scale the input.
    Args:
      min_scale_factor: Minimum scale value.
      max_scale_factor: Maximum scale value.
      step_size: The step size from minimum to maximum value.
    Returns:
        sample: The input sample scaled
    """

    def __init__(self, scale_factors=(0.5, 2.0), discrete=False):
        self.scale_factors = scale_factors
        self.discrete = discrete
        self.mode = {
            'semseg': cv2.INTER_NEAREST,
            'depth': cv2.INTER_NEAREST,
            'normals': cv2.INTER_NEAREST,
            'edge': cv2.INTER_NEAREST,
            'sal': cv2.INTER_NEAREST,
            'human_parts': cv2.INTER_NEAREST,
            'image': cv2.INTER_LINEAR
        }

    def get_scale_factor(self):
        if self.discrete:
            # choose one option out of the list
            random_scale = random.choice(self.scale_factors)
        else:
            assert len(self.scale_factors) == 2
            random_scale = random.uniform(*self.scale_factors)
        return random_scale

    def scale(self, key, unscaled, scale=1.0):
        """Randomly scales image and label.
        Args:
            key: Key indicating the uscaled input origin
            unscaled: Image or target to be scaled.
            scale: The value to scale image and label.
        Returns:
            scaled: The scaled image or target
        """
        # No random scaling if scale == 1.
        if scale == 1.0:
            return unscaled
        image_shape = np.shape(unscaled)[0:2]
        new_dim = tuple([int(x * scale) for x in image_shape])

        unscaled = np.squeeze(unscaled)
        scaled = cv2.resize(unscaled, new_dim[::-1], interpolation=self.mode[key])
        if scaled.ndim == 2:
            scaled = np.expand_dims(scaled, axis=2)

        if key == 'depth':
            # ignore regions for depth are 0
            scaled /= scale

        return scaled

    def __call__(self, sample):
        random_scale = self.get_scale_factor()
        for key, val in sample.items():
            if key == 'meta':
                continue
            sample[key] = self.scale(key, val, scale=random_scale)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PadImage:
    """Pad image and label to have dimensions >= [size_height, size_width]
    Args:
        size: Desired size
    Returns:
        sample: The input sample padded
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = tuple([size, size])
        elif isinstance(size, (list, tuple)):
            self.size = size
        else:
            raise ValueError('Crop size must be an int, tuple or list')
        self.fill_index = {'edge': 255,
                           'human_parts': 255,
                           'semseg': 255,
                           'depth': 0,
                           'normals': [0, 0, 0],
                           'sal': 255,
                           'image': [0, 0, 0]}

    def pad(self, key, unpadded):
        unpadded_shape = np.shape(unpadded)
        delta_height = max(self.size[0] - unpadded_shape[0], 0)
        delta_width = max(self.size[1] - unpadded_shape[1], 0)

        if delta_height == 0 and delta_width == 0:
            return unpadded

        # Location to place image
        height_location = [delta_height // 2,
                           (delta_height // 2) + unpadded_shape[0]]
        width_location = [delta_width // 2,
                          (delta_width // 2) + unpadded_shape[1]]

        pad_value = self.fill_index[key]
        max_height = max(self.size[0], unpadded_shape[0])
        max_width = max(self.size[1], unpadded_shape[1])

        padded = np.full((max_height, max_width, unpadded_shape[2]),
                        pad_value, dtype=np.float32)
        padded[height_location[0]:height_location[1],
            width_location[0]:width_location[1], :] = unpadded
        # else:
        #     padded = np.full((max_height, max_width),
        #                     pad_value, dtype=np.float32)
        #     padded[height_location[0]:height_location[1],
        #         width_location[0]:width_location[1]] = unpadded

        return padded

    def __call__(self, sample):
        for key, val in sample.items():
            if key == 'meta':
                continue
            sample[key] = self.pad(key, val)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomCrop:
    """Random crop image if it exceeds desired size
    Args:
        size: Desired size
    Returns:
        sample: The input sample randomly cropped
    """

    def __init__(self, size, cat_max_ratio=1):
        if isinstance(size, int):
            self.size = tuple([size, size])
        elif isinstance(size, (list, tuple)):
            self.size = size
        else:
            raise ValueError('Crop size must be an int, tuple or list')
        self.cat_max_ratio = cat_max_ratio  # need semantic labels for this

    def get_random_crop_loc(self, uncropped):
        """Gets a random crop location.
        Args:
            key: Key indicating the uncropped input origin
            uncropped: Image or target to be cropped.
        Returns:
            Cropping region.
        """
        uncropped_shape = np.shape(uncropped)
        img_height = uncropped_shape[0]
        img_width = uncropped_shape[1]

        crop_height = self.size[0]
        crop_width = self.size[1]
        if img_height == crop_height and img_width == crop_width:
            return None
        # Get random offset uniformly from [0, max_offset]
        max_offset_height = max(img_height - crop_height, 0)
        max_offset_width = max(img_width - crop_width, 0)

        offset_height = random.randint(0, max_offset_height)
        offset_width = random.randint(0, max_offset_width)
        crop_loc = [offset_height, offset_height + crop_height,
                    offset_width, offset_width + crop_width]

        return crop_loc

    def random_crop(self, key, uncropped, crop_loc):
        if crop_loc is None:
            return uncropped

        cropped = uncropped[crop_loc[0]:crop_loc[1],
                            crop_loc[2]:crop_loc[3], :]
        return cropped

    def __call__(self, sample):
        crop_location = self.get_random_crop_loc(sample['image'])
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_tmp = self.random_crop('semseg', sample['semseg'], crop_location)
                labels, cnt = np.unique(seg_tmp, return_counts=True)
                cnt = cnt[labels != 255]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break
                crop_location = self.get_random_crop_loc(sample['image'])

        for key, val in sample.items():
            if key == 'meta':
                continue
            sample[key] = self.random_crop(key, val, crop_location)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomHorizontalFlip:
    """Horizontally flip the given image and ground truth randomly."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            for key, val in sample.items():
                if key == 'meta':
                    continue
                sample[key] = np.fliplr(val).copy()
                if key == 'normals':
                    sample[key][:, :, 0] *= -1
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize:
    """ Normalize image values by first mapping from [0, 255] to [0, 1] and then
    applying standardization.
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def normalize_img(self, img):
        assert img.dtype == np.float32
        scaled = img.copy() / 255.
        scaled -= self.mean
        scaled /= self.std
        return scaled

    def __call__(self, sample):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        sample['image'] = self.normalize_img(sample['image'])
        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        for key, val in sample.items():
            if key == 'meta':
                continue
            sample[key] = torch.from_numpy(val.transpose((2, 0, 1))).float()
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class AddIgnoreRegions:
    """Add Ignore Regions"""

    def __call__(self, sample):
        for elem in sample.keys():
            tmp = sample[elem]
            if elem == 'normals':
                # Check areas with norm 0
                norm = np.sqrt(tmp[:, :, 0] ** 2 +
                               tmp[:, :, 1] ** 2 + tmp[:, :, 2] ** 2)
                tmp[norm == 0, :] = 255
                sample[elem] = tmp
            elif elem == 'human_parts':
                # Check for images without human part annotations
                if ((tmp == 0) | (tmp == 255)).all():
                    tmp = np.full(tmp.shape, 255, dtype=tmp.dtype)
                    sample[elem] = tmp
            # elif elem == 'depth': # We use 0 as ignore index for depth
            #     tmp[tmp == 0] = 255
            #     sample[elem] = tmp
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PhotoMetricDistortion:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.random() < 0.5:
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.random() < 0.5:
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.random() < 0.5:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.random() < 0.5:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img[:, :, 0] = (img[:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta - 1)) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img

    def __call__(self, sample):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """

        img = sample['image']
        img = img.astype(np.uint8)  # functions need a uint8 image

        # random brightness
        img = self.brightness(img)

        # f_mode == True --> do random contrast first
        # else --> do random contrast last
        f_mode = random.random() < 0.5
        if f_mode:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if not f_mode:
            img = self.contrast(img)

        sample['image'] = img.astype(np.float32)
        return sample

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str
