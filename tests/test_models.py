import pytest
import tensorflow as tf
import numpy as np

# def test_model_lib_exist(get_tf_seg):
#     assert get_tf_seg.models.model_lib
#     assert type(get_tf_seg.models.model_lib) == dict


# def test_model_builder_class(get_tf_seg):
#     for model_builder in get_tf_seg.models.model_lib.values():
#         assert issubclass(model_builder, get_tf_seg.base.model_builder.ModelBuilder)


model_paramters = {
    "unet": {
        "output_size": 1,
        "name": "unet",
        "input_shape": (512, 512, 3),
        "n_filters": [16, 32, 64, 128, 256],
        "activation": "relu",
        "final_activation": "sigmoid",
        "backbone": None,
        "pretrained": "imagenet",
    },
    "deeplabv3plus": {
        "output_size": 1,
        "name": "deeplabv3plus",
        "input_shape": (512, 512, 3),
        "atrous_rates": [6, 12, 18],
        "filters": 8,  # default 256, but 8 is enough for testing
        "activation": "relu",
        "final_activation": "sigmoid",
        "backbone": "ResNet50",
        "pretrained": "imagenet",
    },
}


@pytest.fixture(params=["unet", "deeplabv3plus"])
def get_model_builder(request, get_tf_seg):
    return get_tf_seg.models.model_lib[request.param](**model_paramters[request.param])


def test_models(get_model_builder):
    with tf.device("CPU:0"):
        params = model_paramters[get_model_builder.name]
        model = get_model_builder.build_model()
        output_shape = params["input_shape"][:-1] + (params["output_size"],)

        assert model
        assert isinstance(model, tf.keras.Model)
        assert model.output_shape == (None, *output_shape)
        assert model.input_shape == (None, *model_paramters[model.name]["input_shape"])
        assert model(np.ones(shape=(1, *model_paramters[model.name]["input_shape"]))).shape == (1, *output_shape)
       
def test_backnones():
    pass
    # for backbone in get_tf_seg.models.backbone_lib.values():
    #     assert issubclass(backbone, get_tf_seg.base.backbone.Backbone)  