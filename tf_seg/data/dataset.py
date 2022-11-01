import os
import sys
import time
import urllib
import zipfile

url_camvid = "http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip"
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
    urllib.request.urlretrieve(url, filename="archive.zip", reporthook = report_hook)
    if not os.path.exists("dataset/CamVid"):
        os.mkdir("dataset/CamVid")
    with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
        zip_ref.extractall("dataset/CamVid")
    
    ##These parts will apply the dataset directory processing and stuff
    os.remove("archive.zip")


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
    urllib.request.urlretrieve(url, filename="archive.zip", reporthook=report_hook)
    if not os.path.exists("dataset/MHP"):
        os.mkdir("dataset/MHP")
        with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
            zip_ref.extractall("dataset/MHP")
    
    ##These parts will apply the dataset directory processing and stuff
    os.remove("archive.zip")

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
    urllib.request.urlretrieve(url, filename="archive.zip", reporthook=report_hook)
    if not os.path.exists("dataset/HELEN"):
        os.mkdir("dataset/HELEN")
        with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
            zip_ref.extractall("dataset")

    ##These parts will apply the dataset directory processing and stuff
    os.remove("archive.zip")

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
        download: str = True, # maybe a better parameter name
        init_data_loader: str = False, # maybe a better parameter name
    ):
        pass
    pass

if __name__=="__main__":
    download_camvid_dataset()