import pytest
import tf_seg
from tf_seg.base.model_builder import ModelBuilder

@pytest.fixture
def get_tf_seg():
    return tf_seg
    
@pytest.fixture
def get_base_model_builder():
    return ModelBuilder

@pytest.fixture
def get_source():
    return True

