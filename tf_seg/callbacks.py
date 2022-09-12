"""
Custom callbacks,Check more information in:
- https://github.com/ika-rwth-aachen/PCLSegmentation/blob/main/pcl_segmentation/utils/callbacks.py

"""

import time
from typing import Union, List, Dict
from omegaconf import DictConfig, ListConfig

import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tf_seg.utils import snake_case_to_pascal_case

# from tf_seg.utils.plot import plot_confusion_matrix, confusion_matrix_to_iou_recall_precision, plot_to_image


class MeasureTotalTime(Callback):
    "Measure training time"

    def on_train_begin(self, logs=None) -> None:
        self.start_time = time.time()

    def on_train_end(self, logs=None) -> None:
        self.total_time = time.time() - self.start_time
        print("Total training time: %s" % self.total_time)


class UpdateBestWeights(Callback):
    "Update best weights on end of training"

    def __init__(self, metric_name: str, mode: str) -> None:
        """
        Update best weights on end of training

        Parameters
        ----------
        metric_name : str
            Name of the metric to use for determining the best weights
        mode : str
            One of {min, max}
        """

        super().__init__()

        self.metric_name = metric_name
        self.best_weights = None
        self.best_metric = None

        if mode == "max":
            self.monitor_op = np.greater
            self.best_metric = -np.Inf
        elif mode == "min":
            self.monitor_op = np.less
            self.best_metric = np.Inf
        else:
            raise ValueError("Mode {} not understood".format(mode))

    def on_epoch_end(self, epoch, logs=None) -> None:
        metric = logs[self.metric_name]
        if self.monitor_op(metric, self.best_metric):
            self.best_metric = metric
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None) -> None:
        self.model.set_weights(self.best_weights)


keras_callbacks_lib = {"tensorboard": TensorBoard, "model_checkpoint": ModelCheckpoint, "early_stopping": EarlyStopping, "reduce_lr_on_plateau": ReduceLROnPlateau}
custom_callbacks_lib = {"meausre_total_time": MeasureTotalTime, "uptade_best_weights": UpdateBestWeights}


def _set_callbacks(name: str, config: dict) -> Callback:

    name = config["class_name"]
    params = config["params"]

    if name == "LearningRateScheduler":
        sc_name = config["sc_name"]
        callback = eval(name)(eval(sc_name)(**params))
        return callback
    else:
        callback = eval(name)(**params)
        return callback


def get_callbacks(config: Union[DictConfig, ListConfig], callbacks_lib: Dict[str, Callback] = {**keras_callbacks_lib, **custom_callbacks_lib}) -> Dict[str, Callback]:

    """
    Get callbacks from config file.

    Parameters
    ----------
    config: Union[DictConfig, ListConfig]
        Configuration for training

    Returns
    -------
    callbacks: List[Callback]
        List of callbacks

    """
    pass

    config = config.copy()

    callbacks = {}

    for key, value in config.items():
        # class_name = value["class_name"]
        # params = value["params"]
        c = _set_callbacks(key, value)

        callbacks[key] = c
        # print(snake_case_to_pascal_case(callback))

    return callbacks


# class TensorBoard(tf.keras.callbacks.TensorBoard):
#     """
#     Callback for storing the intermediate results of the model.

#     See Also
#     --------
#     - https://github.com/ika-rwth-aachen/PCLSegmentation/blob/main/pcl_segmentation/utils/callbacks.py

#     """

#     def __init__(self, log_dir, dataset, **kwargs):
#         super().__init__(log_dir, **kwargs)
#         self.dataset = dataset
#         self.num_images = 1
#         self.custom_tb_writer = tf.summary.create_file_writer(self.log_dir + "/validation")

#     def on_train_batch_end(self, batch, logs=None):
#         lr = getattr(self.model.optimizer, "lr", None)
#         steps = self.model.optimizer.iterations
#         with self.custom_tb_writer.as_default():
#             tf.summary.scalar("step_learning_rate", lr(steps), steps)
#         super().on_train_batch_end(batch, logs)

#     def on_epoch_end(self, epoch, logs=None):
#         batch_size = self.model.BATCH_SIZE
#         class_color_map = self.model.CLS_COLOR_MAP

#         # get first batch of dataset
#         (lidar_input, lidar_mask), label, weight = self.dataset.take(1).get_single_element()

#         probabilities, predictions = self.model([lidar_input, lidar_mask])

#         label = label[: self.num_images, :, :]
#         weight = weight[: self.num_images, :, :].numpy()
#         predictions = predictions[: self.num_images, :, :].numpy()

#         # label and prediction visualizations
#         label_image = class_color_map[label.numpy().reshape(-1)].reshape([self.num_images, label.shape[1], label.shape[2], 3])
#         pred_image = class_color_map[predictions.reshape(-1)].reshape([self.num_images, label.shape[1], label.shape[2], 3])
#         weight_image = weight.reshape([self.num_images, weight.shape[1], weight.shape[2], 1])
#         depth_image = lidar_input.numpy()[: self.num_images, :, :, [4]]
#         intensity = lidar_input.numpy()[: self.num_images, :, :, [3]]

#         intensity_image = tf.image.resize(intensity, [intensity.shape[1] * 3, intensity.shape[2] * 3])
#         depth_image = tf.image.resize(depth_image, [depth_image.shape[1] * 3, depth_image.shape[2] * 3])
#         weight_image = tf.image.resize(weight_image, [weight.shape[1] * 3, weight.shape[2] * 3])
#         label_image = tf.image.resize(label_image, [label_image.shape[1] * 3, label_image.shape[2] * 3])
#         pred_image = tf.image.resize(pred_image, [pred_image.shape[1] * 3, pred_image.shape[2] * 3])

#         # confusion matrix visualization
#         figure = plot_confusion_matrix(self.model.miou_tracker.total_cm.numpy(), class_names=self.model.mc.CLASSES)
#         cm_image = plot_to_image(figure)

#         with self.custom_tb_writer.as_default():
#             tf.summary.image("Images/Depth Image", depth_image, max_outputs=batch_size, step=epoch)
#             tf.summary.image("Images/Intensity Image", intensity_image, max_outputs=batch_size, step=epoch)
#             tf.summary.image("Images/Weight Image", weight_image, max_outputs=batch_size, step=epoch)
#             tf.summary.image("Images/Label Image", label_image, max_outputs=batch_size, step=epoch)
#             tf.summary.image("Images/Prediction Image", pred_image, max_outputs=batch_size, step=epoch)
#             tf.summary.image("Confusion Matrix", cm_image, step=epoch)

#         # Save IoU, Precision, Recall
#         iou, recall, precision = confusion_matrix_to_iou_recall_precision(self.model.miou_tracker.total_cm)
#         with self.custom_tb_writer.as_default():
#             for i, cls in enumerate(self.model.mc.CLASSES):
#                 tf.summary.scalar("IoU/" + cls, iou[i], step=epoch)
#                 tf.summary.scalar("Recall/" + cls, recall[i], step=epoch)
#                 tf.summary.scalar("Precision/" + cls, precision[i], step=epoch)

#         super().on_epoch_end(epoch, logs)

#     def on_test_end(self, logs=None):
#         super().on_test_end(logs)

#     def on_train_end(self, logs=None):
#         super().on_train_end(logs)
#         self.custom_tb_writer.close()
