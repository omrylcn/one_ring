import mlflow
import datetime
import tf2onnx

# from knowsmoke.onnx import load_onnx_model


def get_data_info(data_dict) -> dict:
    """
    Get data info from data_dict to use mlflow logging

    """
    assert type(data_dict) == dict, "data_dict must be a dictionary"

    data_info = {}
    data_info["num_examples"] = {}
    data_info["paths"] = {}
    for key, value in data_dict.items():

        # print(f"{key}: {value}")
        data_info["num_examples"][key] = len(value)
        data_info["paths"][key] = value

    return data_info


def log_params(config, log_part=None):
    """
    Log configs  to mlflow
    """
    log_params = log_part if log_part else list(config)  # ["data", "model", "general"]

    for l_param in log_params:
        for k, v in config[l_param].items():
            k = l_param + "_" + k
            mlflow.log_param(k, v)


def log_albumentation(cfg, prefix):

    # assume composee
    tr_cfg = cfg["transform"]["transforms"]

    for t in tr_cfg:
        # Use the transformation class name as part of the parameter name
        class_name = t["__class_fullname__"].split(".")[-1]  # Extracts the class name without the full module path
        param_name = f"{prefix}_{class_name}"

        # Prepare a dictionary with the parameters, excluding '__class_fullname__'
        params = {k: v for k, v in t.items() if k != "__class_fullname__"}

        # Log the parameters of each transformation as a separate parameter, converting the dictionary to a JSON string
        mlflow.log_param(param_name, params)


def log_history(history, initial_epoch=0):  # log_best=None):
    """
    Log history to mlflow

    Parameters
    ----------
    history : dict
        History of training

    log_best : dict
        Best metrics of training
    """

    history_object = history.history if hasattr(history, "history") else history

    for key in history_object.keys():
        epoch = initial_epoch
        for i in range(len(history_object[key])):
            epoch += 1
            mlflow.log_metric(key, float(history_object[key][i]), step=epoch)

        # # last element is max value
        # if log_best:
        #     mlflow.log_metric("best_" + key, float(log_best[key]))




def generate_run_name() -> str:
    """
    Generate a unique run name based on the current datetime.

    The run name is in the format YYYY-MM-DD_HH-MM-SS.

    Returns:
        str: A string representing the current datetime in the format YYYY-MM-DD_HH-MM-SS.
    """

    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")



def initialize_mlflow(experiment_name, run_name):
    """
    Initialize mlflow
    """
    try:
        # active_run = mlflow.active_run()
        # if active_run is None:
        #     mlflow.start_run(run_name=generate_run_name())
        # else:
        #     mlflow.end_run()
        #     mlflow.start_run(run_name=generate_run_name())

        mlflow.set_experiment(experiment_name)
        active_run = mlflow.active_run()

        if active_run is None:
            mlflow.start_run(run_name=run_name)

        elif active_run.info.run_id != run_name:
            mlflow.end_run()
            mlflow.start_run(run_name=run_name)
        else:
            pass

    except Exception as e:
        print(e)


def log_model(model, model_name, use_onnx=True):
    """
    Log model with onnx model optimizer option

    """
    try:
        if use_onnx:
            onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)
            mlflow.onnx.log_model(onnx_model, model_name)
        else:
            mlflow.keras.log_model(model, model_name)
    except Exception as e:
        print(e)
