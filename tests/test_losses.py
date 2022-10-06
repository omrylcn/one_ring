from tf_seg.losses import *
import pytest
import numpy as np

#pred = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10], dtype=np.float32)
#target = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)

pred = np.array([1, 1, 1], dtype=np.float32)
target = np.array([1, 0, 0], dtype=np.float32)


@pytest.fixture(params=__tf_seg_losses__)
def get_loss(request):
    return eval(request.param)()


def test_loss(get_loss):
    assert get_loss
    assert get_loss(pred, target) > 0