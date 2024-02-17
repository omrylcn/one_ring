"""

Constant Parameters, it is not possible change from outside via config file.

"""

# -------------------#
# config part
CONFIG_FILE_EXTENSION = ".yaml"
# constant parameters get from constant.py
CONFIG_STORE_PATH = "./config"
CONFIG_LOAD_STYLE_LIB = [".yaml", ".yml", ".py"]


# -------------------#
# logger part
LOGGER_NAME = "tf_seg"
LOG_FOLDER = "logs"


# -------------------#
# deploy
MODEL_TYPE_LIB = ["tf", "onnx"]
ONNX_PROVIDERS = {"gpu":"CUDAExecutionProvider", "cpu":"CPUExecutionProvider"}
ONNX_OPSET = 13