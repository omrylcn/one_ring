import os
import onnx
import tf2onnx
import tensorflow as tf
from tensorflow.keras.models import Model
from typing import Union


def export_convert(model_name: str, onnx_name: str = "model.onnx", opset: int = 13, optimizer: str = "onnx") -> None:
    """
    Export the model to onnx format

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
    if optimizer == "onnx":
        output_path = os.path.join(model_name, onnx_name)
        try:
            os.system(f"python -m tf2onnx.convert --saved-model {model_name} --output {output_path} --opset {opset}")
        except Exception as e:
            print(f"Error in converting model to onnx : {e}")


def load_onnx_model(onnx_path: Union[str, Model], print_model: bool = False) -> onnx.ModelProto:
    """
    Load onnx model
    Parameters
    ----------
    onnx_path : Union[str,Model]
        onnx model path  or tensorflow.keras.Model
    """
    if isinstance(onnx_path, str):

        onnx_model = onnx.load(onnx_path)
    elif isinstance(onnx_path, Model):
        onnx_model = tf2onnx.convert.from_keras(model, opset=13)
    else:
        raise TypeError("onnx_path should be str or tensorflow.keras.Model")

    onnx.checker.check_model(onnx_model)

    if print_model:
        print(onnx.helper.printable_graph(onnx_model.graph))

    return onnx_model

