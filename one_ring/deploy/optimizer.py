import os
import onnx
import numpy as np
import tf2onnx
from tensorflow.keras.models import Model, load_model
from typing import Union
import onnxruntime as ort


def save_tf_as_onnx(model_name: str, onnx_name: str = "model.onnx", opset: int = 13) -> None:
    """
    Save saved tensorflow model as onnx model

    Parameters
    ----------
    model_name : str
        model name to export
    onnx_name : str, optional
        onnx model name, by default "model.onnx"
    opset : int, optional
        opset version, by default 13
    optimizer : str, optional
        optimizer to use, by default "onnx"

    """
 
    output_path = os.path.join(model_name, onnx_name)
    try:
        os.system(f"python -m tf2onnx.convert --saved-model {model_name} --output {output_path} --opset {opset}")
    except Exception as e:
        print(f"Error in converting model to onnx : {e}")


def save_model_as_onnx(model, onnx_name: str = "model.onnx", opset: int = 13) -> None:
    """
    Convert a Keras model to ONNX and save it to a file.

    Parameters
    ----------
    model : str, tf.keras.Model
        Either a path to a Keras model or a Keras model instance.
    onnx_name : str, default: "model.onnx"
        Path where the converted model will be saved.
    opset : int, default: 13
        The ONNX opset to use for the conversion.
    """

    # Load the model if the input is a path.
    if isinstance(model, str):
        if not os.path.isfile(model):
            raise ValueError(f"No model found at {model}")
        model = load_model(model)

    onnx_model, _ = tf2onnx.convert.from_keras(model, opset=opset)
    onnx.save(onnx_model, onnx_name)


def load_onnx_model(model: Union[str, Model], print_model: bool = False) -> onnx.ModelProto:
    """
    Load onnx model from file or from tf.keras.model

    Parameters
    ----------
    onnx_path : Union[str,Model]
        onnx model path  or tensorflow.keras.Model
    """
    if isinstance(model, str):

        onnx_model = onnx.load(model)
    elif isinstance(model, Model):
        onnx_model = tf2onnx.convert.from_keras(model, opset=13)
    else:
        raise TypeError("onnx_path should be str or tensorflow.keras.Model")

    onnx.checker.check_model(onnx_model)

    if print_model:
        print(onnx.helper.printable_graph(onnx_model.graph))

    return onnx_model


def predict_with_onnx(model, input_data) -> np.ndarray:
    """
    Predict with onnx model

    Parameters
    ----------
    model : onnx.ModelProto
        onnx model
    input_data : np.array
        input data
    
    Returns
    -------
    np.ndarray
        prediction      
    """
    sess = ort.InferenceSession(model.SerializeToString())
    input_name = sess.get_inputs()[0].name
    pred_onx = sess.run(None, {input_name: input_data})
    return pred_onx
