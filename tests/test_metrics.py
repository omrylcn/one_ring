from tf_seg.metrics import *
import pytest
import numpy as np
import tensorflow.keras.backend as K


@pytest.fixture(params=__tf_seg_metrics__)
def get_metric(request):
    return eval(request.param)(name="my_metric")


data_zero = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
data_one = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)

data_pred = np.array([1, 1, 0, 0], dtype=np.float32)
data_target = np.array([1, 1, 1, 0], dtype=np.float32)


def test_metric(get_metric):
    assert get_metric
    assert get_metric.name == "my_metric"

    epsilon = K.epsilon()     
    # assert get_metric(data_zero, data_zero) == np.inf
    # get_metric.reset_states()

    # one score
    assert get_metric(data_one, data_one) == 1
    get_metric.reset_states()

    # zero score 
    assert get_metric(data_zero, data_one) <= epsilon
    get_metric.reset_states()

    assert get_metric(data_one, data_zero) <= epsilon
    #print(get_metric(data_one, data_zero))
    get_metric.reset_states()

    # update state
    get_metric(data_pred, data_target)
    assert get_metric(data_one, data_zero) > epsilon
