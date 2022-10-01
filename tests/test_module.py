import pytest

# test pytest fixture


def test_fixture(get_source):
     assert get_source

def test_tf_seg(get_tf_seg):
    assert get_tf_seg