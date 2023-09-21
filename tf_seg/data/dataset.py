import os
import sys
import time
import urllib
import zipfile

from tf_seg.data import get_data_loader, get_camvid_data_loader # noqa
from tf_seg.config import get_config # noqa
url_camvid = "https://www.kaggle.com/datasets/carlolepelaars/camvid"
url_mhp_v1 = ""     # Need to find the link again.
url_mhp_v2 = ""     # Need to find the link again.
url_helen = "https://pages.cs.wisc.edu/~lizhang/projects/face-parsing/SmithCVPR2013_dataset_resized.zip"


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
    
    """
    if not os.path.exists("dataset/camvid"):
        os.mkdir("dataset/camvid",exist_ok=True)
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
