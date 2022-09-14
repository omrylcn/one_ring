import os

def export_convert(model_name: str, onnx_name: str = "model.onnx", opset: int = 13, optimizer: str = "onnx")-> None:
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



