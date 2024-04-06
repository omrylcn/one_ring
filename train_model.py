import os

# Alternatively, set environment variable (another method)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL messages only

import warnings
import albumentations as A
from datetime import datetime
import argparse

import tensorflow as tf
from tensorflow.keras.metrics import Recall, Precision

import one_ring
from one_ring.config import get_config
from one_ring.data import get_data_loader, get_camvid_data_loader
from one_ring.transformers import Transformer
from one_ring.models import Unet, DeepLabV3Plus, AttUnet
from one_ring.losses import FocalTverskyLoss, DiceLoss, BASNetHybridLoss, JaccardLoss, LogCoshDiceLoss, ComboLoss
from one_ring.train import Trainer
from one_ring.callbacks import get_callbacks
from one_ring.losses import (
    FocalTverskyLoss,
    DiceLoss,
    LogCoshDiceLoss,
    binary_focal_loss,
    categorical_focal_loss,
    FocalLoss,
    sym_unified_focal_loss,
    SymmetricUnifiedFocalLoss,
)
from one_ring.metrics import DiceScore, JaccardScore
from one_ring.callbacks import ORLearningRateCallback
from one_ring.scheduler import ORLearningRateScheduler
from one_ring.utils import generate_overlay_image, calculate_confusion_matrix_and_report, plot_history_dict
from one_ring.utils import set_memory_growth

warnings.filterwarnings("ignore")
tf.get_logger().setLevel(level="ERROR")
set_memory_growth()

print("tensorflow version :", tf.__version__)
print("one_ring version :", one_ring.__version__)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--backbone",
    type=str,
    default=None,
    help="Backbone to use for the model",
)

parser.add_argument(
    "--loss",
    type=str,
    default=None,
    help="Loss function to use for the model",
    choices=["dice_loss","focal_tversky_loss","log_cosh_dice_loss","categorical_focal_loss","sym_unified_focal_loss","symmetric_unified_focal_loss","combo_loss"],
)


args = parser.parse_args()


config = get_config(config_filename="spinal_cord")
config.model.backbone_name = args.backbone if args.backbone else config.model.backbone_name
config.trainer.losses = args.loss if args.loss else config.trainer.losses
print(config.trainer.losses)


tracing_object = {"mlflow": {}}

IM_SIZE = config.data["image_size"][0]
aug_config = {
    "aug_prob": 0.3,
    "random_contrast_limit": 0.2,
    "random_brightness_limit": 0.2,
    "rotate_limit": 10,
}

train_data_loader, val_data_loader = get_data_loader(config.data, train_data=True, val_data=True, test_data=False)
tr_transforms_object = Transformer(config.augmentation, "train").from_dict()
ts_transforms_object = Transformer(config.augmentation, "test").from_dict()
train_dataset = train_data_loader.load_data(transform_func=tr_transforms_object).prefetch(tf.data.AUTOTUNE)
val_dataset = val_data_loader.load_data(transform_func=ts_transforms_object, shuffle=False).prefetch(tf.data.AUTOTUNE)


callbacks = {}
steps_per_epoch = len(train_dataset)
lr_schedule = ORLearningRateScheduler(
    strategy=config.trainer.lr_scheduler["name"],
    total_epochs=config.trainer.epochs,
    steps_per_epoch=steps_per_epoch,
    **config.trainer.lr_scheduler["params"],
).get()


log_dir = f"board_logs/{config.trainer.experiment_name}/" + datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks["tensorboard"] = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, write_images=True, write_graph=True, update_freq="epoch"
)  # , profile_batch='500,520')
callbacks["lr_sch"] = ORLearningRateCallback(lr_schedule)


loss_name = "symmetric_unified_focal_loss"
params = {"gamma": 4 / 3, "alpha": 0.3, "loss_weight": 0.7}

loss_dict = {
    "focal_loss": FocalLoss,
    "dice_loss": DiceLoss,
    "jaccard_loss": JaccardLoss,
    "log_cosh_dice_loss": LogCoshDiceLoss,
    "focal_tversky_loss": FocalTverskyLoss,
    "symmetric_unified_focal_loss": SymmetricUnifiedFocalLoss,
}

loss = loss_dict[loss_name](**params)
metrics = [Recall(), Precision(), JaccardScore()]

model = AttUnet(**config.model).build_model()
#model = Unet(**config.model).build_model()
#model.summary()
#keras.utils.plot_model(model, show_shapes=True, to_file='model.png')
trainer = Trainer(config, model, train_dataset, val_dataset, callbacks=callbacks, metrics=metrics,losses=losses,tracing_object=tracing_object)
# trainer.fit(continue_training=True)
# model = trainer._model