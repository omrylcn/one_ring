"""
Wrapper class module

"""
import tensorflow as tf
from albumentations import Compose, save, load


class AlbumentatiosWrapper:
    """Albumentation Wrapper to use tensorflow dataset."""

    def __init__(self, transforms, output_type=[tf.float32, tf.uint8]):
        """
        Parameters
        ----------
        transforms :  albumentations.Compose
            albumentation transforms
        output_type : list
            a list of output tensorflow type, it is for using to tf.data.Dataset.map function
        """

        self.transforms = transforms
        self.output_type = output_type

    def _aug_albument(self, image, mask):
        data = {"image": image, "mask": mask}
        data = self.transforms(**data)
        image = data["image"]
        image = tf.cast(image, tf.float32)
        mask = data["mask"]
        return image, mask

    @tf.function
    def __call__(self, image, mask):
        aug_img, aug_mask = tf.numpy_function(func=self._aug_albument, inp=[image, mask], Tout=self.output_type)
        return aug_img, aug_mask

    def transform(self, image, mask):
        """Apply augmentation to image and mask"""
        return self.transforms(image=image.numpy(), mask=mask.numpy())

    def save(self, path):
        save(self.transforms, path, data_format="yaml")

    def load(self, path):
        self.transforms = load(path, data_format="yaml")
