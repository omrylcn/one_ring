"""
Albumentations transformers functions for train,valid,test
"""

import albumentations as A


def get_train_transform(image_size: int,p:float=p):
    """
    Albumentations transformers for train data
    """
    train_transforms = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.OneOf(
                [
                    A.RandomSizedCrop(min_max_height=(256, 256), height=image_size, width=image_size, p=p),
                    A.CenterCrop(height=image_size, width=image_size, p=p),
                    A.PadIfNeeded(min_height=image_size, min_width=image_size, p=p),
                ],
                p=1,
            ),
            A.OneOf([A.VerticalFlip(p=p), A.RandomRotate90(p=p), A.Transpose(p=p)]),
            A.OneOf([A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=p), A.GridDistortion(p=p), A.OpticalDistortion(distort_limit=2, shift_limit=p, p=1)], p=0.8),
        ]
    )

    return train_transforms


def get_test_transform(image_size: int):
    """
    Albumentations transformers for test or validation data

    """
    test_transforms = A.Compose(
        [
            A.Resize(image_size, image_size),
        ]
    )

    return test_transforms
