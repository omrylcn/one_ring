"""
See Also
--------
- https://github.com/HasnainRaz/SemSegPipeline/blob/master/dataloader.py

"""

from typing import List, Tuple

import tensorflow as tf
import random
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE


def find_data_extension(path):
    """
    Finds the extension of the data.
    """
    return os.path.splitext(path)[1]


class DataLoader(object):
    """
    A TensorFlow Dataset API based loader for semantic segmentation problems.

    See Also
    --------
    - https://github.com/HasnainRaz/SemSegPipeline/blob/master/dataloader.py

    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        image_size: Tuple[int],
        extensions: Tuple[str] = None,
        channels: Tuple[int] = (3, 1),
        output_type: Tuple[tf.DType] = (tf.uint8, tf.uint8),
        one_hot_encoding: bool = False,
        palette: List[Tuple[int]] = None,
        background_adding: bool = False,
        seed: int = 48,
    ):
        """
        Initializes the data loader object

        Parameters
        ----------
        image : List[str]
            List of paths of train images.
        mask : List[str]
            List of paths of train masks (segmentation masks)
        image_size : Tuple[int]
            Tuple, the final height, width of the loaded images.
        extensions : Tuple[str]
            Tuple, the extensions of the images.
        channels : Tuple[int]
            Tuple of ints, image and mask channels.,
        output_type : Tuple[tf.DType]
            Tuple of tf.DType, the output type of the images and masks.
        pallette : Tuple[int]
            Tuple of ints, the color pallette of the images and masks.
        one_hot_encoding : bool
            If True, one hot encodes the masks.
        background_adding : bool
            If True, adds the background class to the mask.

        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.channels = channels
        self.output_type = output_type
        self.palette = palette
        self.one_hot_encoding = one_hot_encoding
        self.extensions = extensions
        self.background_adding = background_adding

        # check parameters
        if seed is None:
            self.seed = random.randint(0, 1000)
        else:
            self.seed = seed

        # check data ana get decode functions
        self._check_data()

    def _check_data(self):
        """
        Checks the data for errors.
        """
        # TODO add logger check data information process

        # check path
        assert len(self.image_paths) == len(self.mask_paths) and len(self.image_paths) > 0, "Number of images and masks do not match!"

        # check extensions
        if self.extensions is None:

            image_extension = find_data_extension(self.image_paths[0])
            mask_extension = find_data_extension(self.mask_paths[0])
            self.extensions = (image_extension, mask_extension)
            # tf.debugging.assert_equal(sefl, mask_extension, "Image and mask extensions do not match!")
        else:

            image_extension = find_data_extension(self.image_paths[0])
            mask_extension = find_data_extension(self.mask_paths[0])

            assert self.extensions[0] == image_extension, f"Image extensions do not match!, given image extension : {self.extensions[0]} but found : {find_data_extension(self.image_paths[0])}"
            assert self.extensions[1] == mask_extension, f"Mask extensions do not match!, given mask extension : {self.extensions[1]} but found : {find_data_extension(self.mask_paths[0])}"

        self.image_decode_function = getattr(tf.image, "decode_" + self.extensions[0][1:])
        self.mask_decode_function = getattr(tf.image, "decode_" + self.extensions[1][1:])

        # check shape
        image, mask = self._parse_data(self.image_paths[0], self.mask_paths[0])
        # print(image,mask)
        # print(image.shape, mask.shape)
        assert len(image.shape) == 3 and image.shape[-1] == self.channels[0], f"Image has wrong shape!, image shape : {len(image)}"
        assert len(mask.shape) == 3 and mask.shape[-1] == self.channels[1], f"Mask has wrong shape!, mask shape : {len(mask)}"

    def _one_hot_encode(self, image, mask):
        """
        Converts mask to a one-hot encoding specified by the semantic map.
        """
        one_hot_map = []
        for colour in self.palette:
            class_map = tf.reduce_all(tf.equal(mask, colour), axis=-1)
            one_hot_map.append(class_map)
        one_hot_map = tf.stack(one_hot_map, axis=-1)
        one_hot_map = tf.cast(one_hot_map, tf.float32)

        # TODO: add logger information about background class
        # add background if mask is not binary
        if self.background_adding:
            background = 1 - tf.reduce_sum(one_hot_map, axis=-1)
            sh = background.shape
            if len(sh) == 2:
                background = tf.reshape(background, (*sh, 1))
            # background class index is 0
            one_hot_map = tf.concat([background, one_hot_map], axis=-1)

        return image, one_hot_map

    def _parse_data(self, image_path, mask_path):
        """
        Parses the data into a TensorFlow Dataset object.
        Returns:
            A TensorFlow Dataset object.
        """
        image_content = tf.io.read_file(image_path)
        mask_content = tf.io.read_file(mask_path)

        image = self.image_decode_function(image_content, channels=self.channels[0])
        mask = self.mask_decode_function(mask_content, channels=self.channels[1])

        # image = tf.image.decode_png(image_content, channels=self.channels[0])
        # mask = tf.image.decode_png(mask_content, channels=self.channels[0])

        return image, mask

    def _resize_data(self, image, mask):
        """
        Resizes images to specified size.
        """
        image = tf.image.resize(image, self.image_size)
        mask = tf.image.resize(mask, self.image_size, method="nearest")

        return image, mask

    @tf.function
    def _map_fucntion(self, image_path, mask_path):
        """
        Maps the data into a TensorFlow Dataset object.
        """
      
        return tf.py_function(self._sequnce_function, [image_path, mask_path], [self.output_type[0], self.output_type[1]])

    def _create_sequence(self):
        def _sequnce_function(image_path, mask_path):
            image, mask = self._parse_data(image_path, mask_path)
            # print("0image shape :",image.shape)

            if self.one_hot_encoding:
                if self.palette is None:
                    raise ValueError(
                        "No Palette for one-hot encoding specified in the data loader! \
                                    please specify one when initializing the loader."
                    )
                image, mask = self._one_hot_encode(image, mask)

            image, mask = self._resize_data(image, mask)
            # print("1image shape :",image.shape)

            return image, mask

        return _sequnce_function

    def load_data(self, batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
        """
        Loads the data into a TensorFlow Dataset object.
        Parameters
        ----------
        batch_size : int
            the batch size to use.
        shuffle : bool
            if True,shuffle the data.
        seed : int
            if not specified, chosen randomly. Used as the seed for the RNG in the data pipeline.

        Returns
        -------
            A TensorFlow Dataset object.

        """

        self._sequnce_function = self._create_sequence()

        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.mask_paths))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.image_paths), seed=self.seed)

        dataset = dataset.map(self._map_fucntion, num_parallel_calls=AUTOTUNE)
        # dataset = dataset.batch(batch_size)
        # dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset