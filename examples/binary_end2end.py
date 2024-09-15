import os
import datetime
import warnings
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import tensorflow as tf
from tensorflow.keras.metrics import Recall, Precision
from PIL import Image

import one_ring
from one_ring.config import get_config
from one_ring.data import get_data_loader
from one_ring.transformers import Transformer
from one_ring.callbacks import ORLearningRateCallback
from one_ring.scheduler import ORLearningRateScheduler
from one_ring.losses import BoundaryDoULoss
from one_ring.metrics import DiceScore, JaccardScore
from one_ring.models import DeepLabV3Plus
from one_ring.train import Trainer
from one_ring.utils import generate_overlay_image, calculate_confusion_matrix_and_report
from one_ring.deploy.inferencer import Inferencer

warnings.filterwarnings("always")

print(f"TensorFlow version: {tf.__version__}")
print(f"One Ring version: {one_ring.__version__}")

def setup_config():
    config = get_config(config_filename="example_binary")
    IMG_SIZE = config.data["image_size"][0]
    aug_config = {
        "aug_prob": 0.5,
        "rotate_limit": 25,
        "img_size": IMG_SIZE
    }
    return config, aug_config

def get_datasets(config, aug_config):
    train_data_loader, val_data_loader = get_data_loader(config.data, train_data=True, val_data=True, test_data=False)

    train_transforms = A.Compose([
        A.Resize(aug_config["img_size"], aug_config["img_size"]),
        A.HorizontalFlip(p=aug_config["aug_prob"]),
        A.Rotate(limit=aug_config["rotate_limit"], p=aug_config["aug_prob"]),
        A.RandomResizedCrop(height=aug_config["img_size"], width=aug_config["img_size"], scale=(0.95, 1.0), p=aug_config["aug_prob"]),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=aug_config["aug_prob"]),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=aug_config["aug_prob"]),
    ])

    test_transforms = A.Compose([A.Resize(aug_config["img_size"], aug_config["img_size"])])

    tr_transforms_object = Transformer(config.augmentation, "train", train_transforms)
    ts_transforms_object = Transformer(config.augmentation, "test", test_transforms)

    train_dataset = train_data_loader.load_data(transform_func=tr_transforms_object)
    val_dataset = val_data_loader.load_data(transform_func=ts_transforms_object, shuffle=False)

    return train_dataset, val_dataset, {"train": tr_transforms_object.to_dict(), "val": ts_transforms_object.to_dict()}

def setup_callbacks(config, steps_per_epoch):
    callbacks = {}
    lr_schedule = ORLearningRateScheduler(
        strategy=config.train.lr_scheduler["name"],
        total_epochs=config.train.epochs,
        steps_per_epoch=steps_per_epoch,
        **config.train.lr_scheduler["params"]
    ).get()
    callbacks["lr_sch"] = ORLearningRateCallback(lr_schedule)

    log_dir = f"board_logs/{config.train.experiment_name}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks["tensorboard"] = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True, write_graph=True, update_freq='epoch')

    return list(callbacks.values())

def setup_loss_and_metrics():
    loss = BoundaryDoULoss(gamma=4/3, alpha=0.4, loss_weight=0.4)
    metrics = [Recall(), Precision(), JaccardScore(), DiceScore()]
    return loss, metrics

def train_model(config, train_dataset, val_dataset, callbacks, metrics, loss, save_object):
    model = DeepLabV3Plus(**config.model).build_model()
    trainer = Trainer(config, model, train_dataset, val_dataset, callbacks=callbacks, metrics=metrics, loss=loss, save_object=save_object)
    history = trainer.fit()
    trainer.finalize_training()
    return trainer, history

def plot_history(history, figsize=(20, 15)):
    n_metrics = len(history['metrics']) + 1
    n_rows = (n_metrics + 1) // 2
    
    fig, axs = plt.subplots(n_rows, 2, figsize=figsize)
    fig.suptitle('Model Training History', fontsize=16)
    axs = axs.flatten()
    
    axs[0].plot(history['loss'], label='Train')
    axs[0].plot(history['val_loss'], label='Validation')
    axs[0].set_title('Model Loss')
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    for i, metric in enumerate(history['metrics'].keys(), start=1):
        axs[i].plot(history['metrics'][metric], label='Train')
        axs[i].plot(history['val_metrics'][metric], label='Validation')
        axs[i].set_title(metric.capitalize())
        axs[i].set_ylabel(metric.capitalize())
        axs[i].set_xlabel('Epoch')
        axs[i].legend()
        axs[i].grid(True, linestyle='--', alpha=0.7)
    
    axs[-1].plot(history['epoch_time'])
    axs[-1].set_title('Epoch Time')
    axs[-1].set_ylabel('Time (seconds)')
    axs[-1].set_xlabel('Epoch')
    axs[-1].grid(True, linestyle='--', alpha=0.7)
    
    for i in range(n_metrics + 1, len(axs)):
        fig.delaxes(axs[i])
    
    plt.tight_layout()
    plt.show()
    
    print(f"Final training loss: {history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    
    for metric in history['metrics'].keys():
        print(f"Final training {metric}: {history['metrics'][metric][-1]:.4f}")
        print(f"Final validation {metric}: {history['val_metrics'][metric][-1]:.4f}")
    
    print(f"Average epoch time: {np.mean(history['epoch_time']):.2f} seconds")

def evaluate_model(model, val_dataset, threshold=0.6):
    for i in val_dataset:
        pred_logits = model.predict(i[0])
        
        for n in range(len(i[0])): 
            image = i[0][n].numpy().astype(np.uint8)
            pred_logit = pred_logits[n]
            pred_value = np.where(pred_logit > threshold, 1, 0).reshape(224, 224, 1)
            pred_mask = (pred_value * 255).astype(np.uint8)

            overlay = generate_overlay_image(pred_mask, image, alpha=0.3)
            target = i[1][n].numpy()

            plt.figure(figsize=(20, 10))
            plt.subplot(1, 3, 1)
            plt.title('target')
            plt.imshow(target)
            plt.grid(True)
            plt.subplot(1, 3, 2)
            plt.title('pred')
            plt.imshow(pred_value)
            plt.grid(True)
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.grid(True)
            plt.show()

            cm, cr = calculate_confusion_matrix_and_report(pred_value, target)
            print(cm)
            print(cr)

def setup_inferencer(model_path):
    config = {
        "model_type": "onnx",
        "model_path": model_path,
        "preprocessor_type": "albumentations",
        "postprocessor_type": "vanilla",
        "device": "cpu",
        "threshold": 0.6
    }
    return Inferencer(config, cache_size=256, log_level="INFO")

def main():
    config, aug_config = setup_config()
    train_dataset, val_dataset, save_object = get_datasets(config, aug_config)
    callbacks = setup_callbacks(config, len(train_dataset))
    loss, metrics = setup_loss_and_metrics()
    
    trainer, history = train_model(config, train_dataset, val_dataset, callbacks, metrics, loss, save_object)
    plot_history(history)
    
    evaluate_model(trainer.model, val_dataset)
    
    best_model_path = trainer.save_path
    print(f"Best model path: {best_model_path}")
    
    inferencer = setup_inferencer(best_model_path)
    
    # Example inference
    image_path = 'data/vessel/val_images/1.png'
    image = np.array(Image.open(image_path))
    pred_mask, pred_image = inferencer(image)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(pred_mask[0])
    plt.title("Predicted Mask")
    plt.subplot(1, 2, 2)
    plt.imshow(pred_image[0])
    plt.title("Predicted Image")
    plt.show()

if __name__ == "__main__":
    main()