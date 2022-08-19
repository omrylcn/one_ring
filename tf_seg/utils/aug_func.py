import tensorflow as tf
import albumentations as A


class AlbumentatiosWrapper:
    """Albumentation Wrapper to use tensorflow dataset."""

    def __init__(self, transforms, output_type=[tf.float32, tf.float32]):
        self.transforms = transforms

    def _aug_albument(self, image, mask):
        data = {"image": image, "mask": mask}
        data = self.transforms(**data)
        image = data["image"]
        mask = data["mask"]
        return image, mask

    @tf.function
    def __call__(self, image, mask):
        aug_img, aug_mask = tf.numpy_function(func=self._aug_albument, inp=[image, mask], Tout=[tf.float32, tf.float32])
        return aug_img, aug_mask
