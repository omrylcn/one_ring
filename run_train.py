import os
import warnings
from datetime import datetime
import tensorflow as tf
import albumentations as A
from one_ring.config import get_config
from one_ring.data import get_data_loader
from one_ring.transformers import Transformer
from one_ring.models import DeepLabV3Plus
from one_ring.train import Trainer
from one_ring.losses import (
    FocalLoss, DiceLoss, JaccardLoss, LogCoshDiceLoss,
    FocalTverskyLoss, SymmetricUnifiedFocalLoss, BoundaryDoULoss
)
from one_ring.metrics import JaccardScore
from one_ring.callbacks import ORLearningRateCallback
from one_ring.scheduler import ORLearningRateScheduler
from one_ring.utils import set_memory_growth
from one_ring.deploy.inferencer import Inferencer

import matplotlib.pyplot as plt
import logging

# Suppress warnings and set logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
warnings.filterwarnings('ignore')
tf.get_logger().setLevel("ERROR")

set_memory_growth()

def setup_augmentations(image_size, aug_prob=0.1):
    """Set up augmentation pipelines for training and testing."""
    train_transforms = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=aug_prob),
        A.VerticalFlip(p=aug_prob),
        A.Rotate(limit=10, p=aug_prob),
        A.RandomSizedCrop(min_max_height=(180, 224), height=image_size, width=image_size, p=aug_prob),
    ])
    test_transforms = A.Compose([A.Resize(image_size, image_size)])
    return train_transforms, test_transforms

def setup_callbacks(config, steps_per_epoch):
    """Set up callbacks for the training process."""
    callbacks = {}
    lr_schedule = ORLearningRateScheduler(
        strategy=config.train.lr_scheduler["name"],
        total_epochs=config.train.epochs,
        steps_per_epoch=steps_per_epoch,
        **config.train.lr_scheduler["params"]
    ).get()
    callbacks["lr_sch"] = ORLearningRateCallback(lr_schedule)
    
    log_dir = f"board_logs/{config.train.experiment_name}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    callbacks["tensorboard"] = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_images=True, write_graph=True, update_freq='epoch'
    )
    return list(callbacks.values())

def main():
    # Load configuration
    config = get_config(config_filename="spinal_cord")
    
    # Prepare data
    train_data_loader, val_data_loader = get_data_loader(config.data, train_data=True, val_data=True, test_data=False)
    image_size = config.data["image_size"][0]
    train_transforms, test_transforms = setup_augmentations(image_size)
    
    tr_transforms_object = Transformer(config.augmentation, "train", train_transforms)
    ts_transforms_object = Transformer(config.augmentation, "test", test_transforms)
    
    train_dataset = train_data_loader.load_data(transform_func=tr_transforms_object)
    val_dataset = val_data_loader.load_data(transform_func=ts_transforms_object, shuffle=False)
    
    # Setup callbacks
    callbacks = setup_callbacks(config, len(train_dataset))
    
    # Prepare loss and metrics
    loss_dict = {
        "focal_loss": FocalLoss,
        "dice_loss": DiceLoss,
        "jaccard_loss": JaccardLoss,
        "log_cosh_dice_loss": LogCoshDiceLoss,
        "focal_tversky_loss": FocalTverskyLoss,
        "symmetric_unified_focal_loss": SymmetricUnifiedFocalLoss,
        "boundary_dou_loss": BoundaryDoULoss
    }
    loss_name = "boundary_dou_loss"
    loss_params = {"gamma": 4/3, "alpha": 0.4, "loss_weight": 0.4}
    loss = loss_dict[loss_name](**loss_params)
    metrics = [tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), JaccardScore()]
    
    # Prepare model and trainer
    model = DeepLabV3Plus(**config.model).build_model()
    save_object = {
        "transformer": {
            "train": tr_transforms_object.to_dict(),
            "val": ts_transforms_object.to_dict()
        }
    }
    trainer = Trainer(config, model, train_dataset, val_dataset, callbacks=callbacks, metrics=metrics, loss=loss, save_object=save_object)
    
    # Train the model
    history = trainer.fit()
    trainer.finalize_training()
    save_path = trainer.save_path
    
    # Setup inferencer
    inferencer_config = {
        "model_type": "onnx",
        "model_path": save_path,
        "preprocessor_type": "albumentations",
        "postprocessor_type": "vanilla",
        "device": "cpu",
        "threshold": 0.6
    }
    inferencer = Inferencer(inferencer_config, cache_size=256, log_level="INFO")
    
    # Test inference
    image_path = "examples/test_images/road_2.jpg"
    image = plt.imread(image_path)
    # plt.imshow(image)
    # plt.show()
    
    # Perform inference (you may want to add this part)
    pred_image, pred_mask = inferencer(image)
    # plt.imshow(pred_mask)
    # plt.show()

if __name__ == "__main__":
    main()