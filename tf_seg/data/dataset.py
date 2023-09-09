import os
import sys
import time
import urllib
import zipfile
from datasets import load_dataset

from tf_seg.data import get_data_loader, get_camvid_data_loader # noqa
from tf_seg.config import get_config # noqa

#TODO: Load the datasets in huggingface_hub and set the data api.
url_camvid = "https://huggingface.co/datasets/yesilyurt/CamVid"
url_mhp_v1 = """https://doc-14-3o-docs.googleusercontent.com/docs/securesc/qvdk5vi1ibqiblcvk7b5kb0iunlth1ks/bgha1ubbaieguqk3ku83i0oipa18n950/1694260650000/14300881397912700555/10737577685452255707/1h
    TS8QJBuGdcppFAr_bvW2tsD9hW_ptr5?e=download&ax=AH3YgiDGhumf0-FuXBe3k0tqVeES0nJ5jSXhgdwTDLdbHfzf_NqPUEf1NpXTPGUzymO_FJkx3K8EsT4J4-yqlmRUglp0H4YODpvj5MWJ1-Wewdy3sXAmTtYprIg-9882LD4dujeX1MijkRKWq0YB0_9O0O
    euxa7qvKmgSCF-bbnlm86iblB_YwlLTTpQBknSfi9-M9Q-UtIQcFAiLJu6Qc72w_BWrlCaMXQMK2c2MYBhDgJCmNgPJifyLI7FaGtU8JD-m5kpR0CtM4h3gxNepnd5E-eXMkkYdXjJsHSdNeWUH5KjmrxhIjJob41B4KfhWduNbhg3YzBvTqXAKuc9Xl4Wwyw1G_7SjN
    Fn_iG-apVmYbCSPW4UR8V9VzEJzcl8stmSyB6A9CdrMAcd1ElPDhOFZ0ShFcPLY5n90Yf_qiqpDxaNwV8lZXl47azC57j0Wz1-gXCs9M188sTxOTqIRu69SkSQSpOvRTGXN9KHAxDzbfrk9p92N5FDKJdqSZRDtLvPNGPQqQiOk2IAVQrI7MDr5H574
    -IJQNKdTggEdRgiYxWBa9ICYYzTa2LD4p9MhAag1c92JmsfcsCKxieOsnqjyK8Zdiqa69Mu3vrCdvrlNmgu5Z2pR10ySg_8FcxqILO9crf42GZbvFwIbjcJuk7835GZgHCNYxtPYjCYYeuG3cQiOdlASWnLUID_C_abaPNruAs27b2mdUhsv-
    Gt7qZ2HxCZoy21T6iPVbWuQv7dW23sMwLOPqQf24q0np7dWTLXmeFycrMXIsyqyBf3rsMc4NXgcl5_2LxfJft5QCr7b2vurjbWlL7nrG76i54DwKMUFyycAHl1y9ETfqRlZbhXDwq0KVqugote47wr0AZFVruVKrKxyGMnIaXfdAYQHfw
    &uuid=f0477e42-b878-4e84-a024-31b642801194&authuser=0"""     # Need to find the link again.
url_mhp_v2 = """https://doc-08-3o-docs.googleusercontent.com/docs/securesc/qvdk5vi1ibqiblcvk7b5kb0iunlth1ks/j854o7p9bjfs4j8jjspg0gp1tkp3ol3l/1694260575000/14300881397912700555/10737577685452255707
    /1YVBGMru0dlwB8zu1OoErOazZoc8ISSJn?e=download&ax=AH3YgiCnE8UcGIqp_ba_0CYHZ_7kX6_N_9sKKuE5EcgmMEJXw0SI49-
    CeG5GM_q6jiFuRvguDKp8gi3ChMlttBwZLu4Wjaqf5c5PEwrIM_NJrcDOr3Bao_aHtGqJux2C9qVx4aBg62D659seKQSRwbpJZ2TZ2TyZpHlyRZevlfyCik1eyPcdXlPu1HR1D3omXdkyC0f55cOF707iHV09CEnaWyJMkeSSBFy9NxSYfoaEI2luxppSEcWnAVfXwp3
    YY8LTiuuKiPIvFkCXaUp6l_NB6xwm6NbuTJvNIfYgqg-IjQOSEmtR0j8t3-y1BHDANB5sHKEGuINSRfPNmf0mORgkP8yhQOh3S3jShdVybvii96LIIJg_TPgSclMBZOJ2OoSYUESenmpxbginZiFoXLl9VpTN_BM7r104C5AT3muVmlXWaalk
    -aygMCZKYjKSGZ4j3VttFCmvOmaVKEUSgwffsNlG8PIdRviBAOqd7WHqn42ROlpQsDLtbCWxTXCQySA7I0VK4UR3HczDlsaV9bfqltIlt-TrZX4-
    MVnmcjmlvigtXduqtJWu3S28a6t2qUIwAcwO6GQTNyBOILtm8hpy_aG9uJBjlK66akSgkUhYIoupOa8P89hn4od00hXc1QUoRQ812L2dgi90V_yxFvp5QsAXgEPucjzn7pZEKD-CKBS-iRdd-
    cfI4vHgXDXGULydGIuhVnMwFRTE1zaOCTFmMdD8SFdOJKiuO1P8_sFPFTHPaiJT6UlRbSUNrSysdXR6M6S7yXUA-VMteChOduSu8YKhvybG7xsO_SWXNqJLm2SfRmJmt7OgUoXnq8nkMQb0V2t_bpNFc4QuB4PBVmy6JmwSZ4UvYXAVaRDIu7b3dMu2NqjuJHmcW1
    -R3AgFRLo&uuid=4f004700-aa40-4bda-a5f8-969c28299eeb&authuser=0&nonce=upsmsjpo7tiai&user=10737577685452255707&hash=rninfgfoch6oasvhkr3vmnuqtc42vhp1"""     # Need to find the link again.
url_helen = "https://pages.cs.wisc.edu/~lizhang/projects/face-parsing/SmithCVPR2013_dataset_resized.zip"
hf_datasets = ["CamVid"]


def get_huggingface_datasets():
    """Prints the list of huggingface datasets that can be used within this module.

    """
    print(hf_datasets)


def get_dataset_from_huggingface(dataset_name: str):
    """Returns the dataset given its name.

    Parameters
    ----------
    dataset_name: str
        The name for the dataset to be pulled fom huggingface_hub. The list of supported datasets can be viewed via the function get_huggingface_datasets().
    
    """
    assert dataset_name in hf_datasets, "The dataset you are looking is not supported by huggingface data API."
    dataset = load_dataset(dataset_name)
    return dataset


def report_hook(count, block_size, total_size):     # Need to add downloading speed feature.
    start_time = time.time()                        # Need also to fix progress_size.
    if count == 0:
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    # speed = progress_size / (1024.**2 * duration)
    percent = count * block_size * 100. / total_size
    sys.stdout.write("\r%d%% | %d MB | %d sec elapsed" %
                     (percent, progress_size / (1024.**2), duration))
    sys.stdout.flush()


def download_camvid_dataset(url: str = url_camvid):
    """ Downloads the CamVid dataset.

    Parameters
    ----------
    url: str, default=url_camvid
        The url to get the data from, by default url_camvid as defined above.

    Notes
    -----


    References
    ----------
    CamVid Dataset on Kaggle: https://www.kaggle.com/datasets/carlolepelaars/camvid
    """
    if not os.path.exists("dataset/camvid"):
        os.mkdir("dataset/camvid")
    urllib.request.urlretrieve(url, filename="dataset/camvid/archive.zip", reporthook=report_hook)
    with zipfile.ZipFile("dataset/camvid/archive.zip", 'r') as zip_ref:
        zip_ref.extractall("dataset/camvid")
    
    ##These parts will apply the dataset directory processing and stuff
    os.remove("dataset/camvid/archive.zip")


def download_mhp_dataset(url: str = url_mhp_v2):
    """ Downloads the MHP_V2 dataset.

    Parameters
    ----------
    url: str, default=url_mhp_v2
        The url to get the data from, by default the url_mhp_v2 as defined above.

    Notes
    -----

    References
    ----------

    """
    if not os.path.exists("dataset/mhp"):
        os.mkdir("dataset/mhp")
    urllib.request.urlretrieve(url, filename="dataset/mhp/archive.zip", reporthook=report_hook)
    with zipfile.ZipFile("dataset/mhp/archive.zip", 'r') as zip_ref:
        zip_ref.extractall("dataset/mhp")
    
    ##These parts will apply the dataset directory processing and stuff
    os.remove("dataset/mhp/archive.zip")


def download_helen_dataset(url: str = url_helen):
    """ Downloads the HELEN dataset.

    Parameters
    ----------
    url: str, default=url_helen
        The url to get the data from, by default the url_helen as defined above.

    Notes
    -----

    References
    ----------

    """
    if not os.path.exists("dataset/helen"):
        os.mkdir("dataset/helen")
    urllib.request.urlretrieve(url, filename="dataset/helen/archive.zip", reporthook=report_hook)
    with zipfile.ZipFile("dataset/helen/archive.zip", 'r') as zip_ref:
        zip_ref.extractall("dataset/helen")

    ##These parts will apply the dataset directory processing and stuff
    os.remove("dataset/helen/archive.zip")


def download_data(data_name: str):
    """ Downloads the dataset specified as parameter.

    Parameters
    ----------
    data_name: {"camvid", "mhp", "helen"}
        The dataset name to download.
    """
    downloader = globals()[f"download_{data_name}_dataset"]
    downloader()


class DataLib(object):  # Better name maybe
    """ Creates a dataset instance given the dataset parameter as the dataset name.

    Parameters
    ----------
    dataset: {"camvid", "mhp", "helen"}
        The dataset name to create instance of it.
    download: bool, default=True
        If the dataset is not found locally, download it. Default is True.
    init_data_loader: bool, default=False
        Initiates a DataLoader instance of tf_seg.data.DataLoader. Default is False.

    Examples
    --------
    
    """
    def __init__(
        self,
        dataset: str,
        download: bool = True,  # maybe a better parameter name
        init_data_loader: bool = True,  # maybe a better parameter name
    ):
        super.__init__()
        pass
    pass


class CamvidDataset(DataLib):
    def __init__(
        self,
        download: bool = True,
        init_data_loader=True,
    ):
        super.__init__(dataset="camvid",
                       download=download,
                       init_data_loader=init_data_loader)
        if init_data_loader:
            self.loader = get_camvid_data_loader(data_config="config.yaml")

    def __call__():
        pass

    
def get_camvid(init_data_loader: bool = True):
    ''' Returns an instance of the CamVid dataset

    Parameters
    ----------
    '''

    if not os.path.exists("dataset/camvid"):
        dataset = CamvidDataset(init_data_loader=init_data_loader)
    else:
        dataset = CamvidDataset(download=False, init_data_loader=init_data_loader)

    return dataset
