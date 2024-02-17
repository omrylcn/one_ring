"""
This code borrowed from : https://github.com/ika-rwth-aachen/PCLSegmentation/blob/main/pcl_segmentation/utils/util.py

"""

import io
import itertools

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix



def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Parameters
    ----------
    cm :(array, shape = [n, n])
        a confusion matrix of integer classes
    class_names (array, shape = [n]):
        String names of the integer classes

    """
    # normalize confusion matrix
    cm_normalized = np.around(
        cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2
    )

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm_normalized[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def confusion_matrix_to_iou_recall_precision(cm):
    """
    Computes the classwise iou, recall and precision from a confusion matrix
    cm: Confusion matrix as nxn matrix where n is the number of classes
    Confusion matrix has switched axes when taken from *total_cm* !!
    """
    with tf.name_scope("compute_iou_recall_precision") as scope:
        sum_over_col = tf.reduce_sum(
            cm, axis=1
        )  # axes are switched for the total_cm within MeanIoU
        sum_over_row = tf.reduce_sum(
            cm, axis=0
        )  # axes are switched for the total_cm within MeanIoU
        tp = tf.linalg.diag_part(cm)
        fp = sum_over_row - tp
        fn = sum_over_col - tp
        iou = tf.math.divide_no_nan(tp, tp + fp + fn)
        recall = tf.math.divide_no_nan(tp, tp + fn)
        precision = tf.math.divide_no_nan(tp, tp + fp)
    return iou, recall, precision


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())




# def generate_overlay_image(pred_image, input_image, alpha=0.3):
#     """
#         Generate an overlay image by applying a green mask to the input image based on the prediction mask.

#         Parameters
#         ----------
#         pred_image : np.ndarray
#             The prediction image as a numpy array. This can be a binary mask or any array where values greater than 0.5 are considered part of the mask.
#         input_image : np.ndarray
#             The original image on which the overlay will be applied. Must be the same dimensions as `pred_image`.
#         alpha : float, optional
#             The transparency factor of the overlay. Higher values make the green mask more opaque. Defaults to 0.3.

#         Returns
#         -------
#         np.ndarray
#             The resulting overlay image as a numpy array with the same shape as `input_image`.

#         Raises
#         ------
#         AssertionError
#             If `pred_image` or `input_image` is not a numpy array or if their dimensions do not match.

#         Examples
#         --------
#         >>> input_img = cv2.imread("path/to/image.png")
#         >>> pred_img = np.random.rand(*input_img.shape[:2])  # Dummy prediction data
#         >>> overlay_img = generate_overlay_image(pred_img, input_img, alpha=0.5)
#         >>> cv2.imshow("Overlay", overlay_img)
#         >>> cv2.waitKey(0)

#         Notes
#         -----
#         - The function assumes that `pred_image` has the same height and width as `input_image`.
#         - The `alpha` parameter controls the blend between the original image and the green mask. An `alpha` of 1.0 would result in only the green mask being visible.

#     """
#     # Validate inputs
#     assert isinstance(pred_image, np.ndarray), f"pred_image must be a numpy array, got {type(pred_image)}"
#     assert isinstance(input_image, np.ndarray), f"input_image must be a numpy array, got {type(input_image)}"
#     assert pred_image.shape[:2] == input_image.shape[:2], "pred_image and input_image must have the same dimensions"

#     # Ensure pred_image is boolean to create a mask, and apply threshold if necessary
#     if pred_image.dtype != np.bool_:
#         pred_mask = np.where(pred_image > 0.5, 1, 0).astype(bool)
#     else:
#         pred_mask = pred_image

#     # Create a green color mask where the prediction is positive
#     colored_mask = np.zeros_like(input_image)
#     colored_mask[pred_mask, 1] = 255  # Set green channel to max for masked areas

#     # Apply the colored mask to the input image with the specified alpha for the overlay
#     overlay = cv2.addWeighted(input_image, (1-alpha), colored_mask, alpha, 0)

#     return overlay

def generate_overlay_image(pred_image, input_image, alpha=0.3):

    """
    Generate an overlay image by applying a green mask to the input image based on the prediction mask.

    Parameters
    ----------
    pred_image : np.ndarray
        The prediction image as a numpy array. This can be a binary mask or any array where values greater than 0.5 are considered part of the mask.
    
    """    


    assert type(pred_image) == np.ndarray, f"pred_image should be a numpy array,{type(pred_image)}"
    assert type(input_image) == np.ndarray, f"input_image should be a numpy array,{type(input_image)}"

    pred_mask = pred_image.astype(np.uint8)
    image = input_image.astype(np.uint8)

   
    # masked = cv2.bitwise_and(image, image, mask=pred_mask)
    
    colored_mask = np.zeros_like(image)
    #colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    # colored_mask = cv2.merge((pred_mask,pred_mask,pred_mask))
    
    colored_mask[:, :, 1] = pred_mask[:, :, 0]
    overlay = cv2.addWeighted(image, (1-alpha), colored_mask, alpha, 0)

    return overlay



def calculate_confusion_matrix_and_report(pred, target):
    """
    Calculate the confusion matrix and classification report for the given predictions and target values.

    Parameters
    ----------
    pred : np.ndarray
        The predicted labels as a numpy array. Expected shape is (n_samples,) after reshaping.
    target : np.ndarray
        The true labels as a numpy array. Expected shape is (n_samples,) after reshaping.

    Returns
    -------
    tuple
        A tuple containing two elements:
        - The confusion matrix as a numpy array of shape (n_classes, n_classes).
        - The classification report as a string.

    """
    # Ensure predictions and targets are 1D arrays
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    # Calculate the confusion matrix and classification report
    cm_result = confusion_matrix(target, pred)
    cr_result = classification_report(target, pred, zero_division=0)

    return cm_result, cr_result


