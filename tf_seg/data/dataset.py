import os
import urllib
import zipfile

url_camvid = "https://storage.googleapis.com/kaggle-data-sets/635428/1132317/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20221006%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20221006T152140Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=2c7e207916903eafc1e58493d3b54304b271a1e6699796219be1de5e7b4f919da977d244432b3b1b0c4539e3c732de920f572028bbb923cae8b12a87f03acc6d4dbf0d6778dd62acaf4a54e39112f1c573316666128591073b34756c441cbc95244ec86df3c69f5ab0eee10ea744776a95f66a2e4e0e6ed01050529c3baa48886fbdbd1495e6b9f392adb9ea1f5d02ed5b58bacf73d8055f4795c914766a8ee85615a575e73fd628baddab50b7b205e8eba31ec5ab6a91322b818ed408bb09b55ad384098c58f988c4c385106793889ebb1bfc1fca234c2caecf35f5b7861f650c2d14ec90ebd7f49c9a2d300b40fa95ba6ce552fcf2eb874b164934ec0c15c4"
url_mhp_v1 = "https://doc-14-3o-docs.googleusercontent.com/docs/securesc/qvdk5vi1ibqiblcvk7b5kb0iunlth1ks/k731pk9v6rqj51g99kih0euukmd11711/1665089850000/14300881397912700555/10737577685452255707/1hTS8QJBuGdcppFAr_bvW2tsD9hW_ptr5?e=download&ax=ALW9-sDAuW1q0GAlw3Hcu6K_GM2CZTlTLzZ35b3gCb1B7ugxffHqf9H7_A9YMWMKEErl7QzkIXbpGv0bxasZG25yPkIfaLVOH06skcEifL0hsEN6SbNqFPmDsVm2VgFwvnS9A1BAh31Uo7wm_i1up7W7LnndICLZocYTQ3GJH6dguYrhOnPWilzfBXnJRhZbbeKo_ZCEJ-KEpr2VzKQAU-8z1rilohZU1H6gkoJHrSp95p-6zfFYTJcsItq-rAxq49knkc26q2M7fnA3Bym9IHxIvpJq1eFu_waQnFsHGiQzErM-4LoNx2oVFXnCKYJGS5E3bZppEOZcoXWNUYUs4N3qn31m-6bKBcQA2QXSHmLj4yuRQ2KnyQLAHKh5WhH3R5HtmbMhn-q5rt8_jx6ZBAh9e-_cku2-8grvDBmvhN4_rEBJqBZBQALxP7RyJYimvL9kHM4viU5ApUJa3YC98hHF920hZqY_n0ZKSFX1rSmj75cG9ajRFUnErKPU2rHiqCUZqv1crncDo20Szj_8b_jBFQkI_BVGpHT_uxUJOzYObJPKCeW3wfVGgsVke1lFA7ecwqYTV1h7T17XIG_UFEKIhOYUP7GIbUdJgC9K4pBOEP-rKGjC9gz7WWVzi6gPZOGZIf1HSBNMEQJy_nJ8wxYvQcD7YpTsrpO3B3-Tg089yVu9b-GjGg9PPhev1hCRLmirDeZmFv7r5pIDIoAa07pXlKTxNn6CcsCNGe9uR0fuDRY5psJ0mIkmeb1b5fSe1KN5oqxriCd5WqGhaqolcrIwG0L7vnr2pdr-0BaaosIjSqh2tiRQmWrBqYSjKlcFz1Yrw56KTseHsmDoN4uYWTnsGiflyyJT4v1ZAyW3yFGde_3GFLCcwU7sEAcUL4BukpDyZCttT89lOVONpVc3YK3IzqJQNcsNAg&uuid=844442b3-af43-4715-8e36-a228c3fb7f96&authuser=0&nonce=dck44bqga04a6&user=10737577685452255707&hash=e5k2dcd3pb4gnjtk42m5dh78pb7m3lns"
url_mhp_v2 = "https://doc-08-3o-docs.googleusercontent.com/docs/securesc/qvdk5vi1ibqiblcvk7b5kb0iunlth1ks/4fjb4akeu8crm2r91bpbc5qlpgct0fp4/1665089925000/14300881397912700555/10737577685452255707/1YVBGMru0dlwB8zu1OoErOazZoc8ISSJn?e=download&ax=ALW9-sAsZhngdqL3YkK-9oTuz0O07JCLblyc0TWRiU0j9INi9lPYY2vxHClg119xtUm1knhcYo7GrHRWt1wdiNnYHnjqrUv5TUmQn7dum1RBJIs_ljNeSI8RH9mzPO2gJMc-lPw6h3SBV3HuX4UYuZYGY1XJBy3wLP1YSRHtaNarQuA5jAXtlMiYJ42bJsCGKob0SHxASD8kQ-YcvMYww6-nJ-KMw5WftfE7pT4WAvWgXR-TxF2jx6YaeSBy-m-T3CjFI4J20jHWFTL6IKLkFWP3rt7sKqhAV3LC93QPU0yu7dQdX5-kDj6MaxWkqAvWOyylldY0Xj9NtkbAOMkpwTPbsKfVl17Rtofg6NMnUyxetF4c6yDyEJXPgH1qkZWyLwMpKcfO4lK7LxvY94-9ZO2H_ce57VO3ZRpuBIOi6T8HjQbRahB5XQaTBE9Et9zv5cOMUywtG4ORc46msoYP3gvnD8sR58D1y9j9tlr8uluVk2De8E0hS3b4zCzNzPvVGk3E5wXDyeqtfgopOf29l8TkGNsbPrG7MoPs4W8MjCKRuBBQyX4tlLfkrLRUFrSdF6CTh7Q6CJ-1T2V3k26BQ1y73we9gkUGfZEm7k5YYRAELLF1DEnwWQI4rIXUMvoS0nF32zxe27yx8NGhqomfvDv1wCn24DENgjBpQsrICBqrXd-HMbuXtGk3HRpmO1YO-zCUe2N_deBEeZp669Mj6XsDQnLH8v-hjUiNSFdlC48psNJuRPRutYh2Y_1kzByNx4_bNNF2tUUgAU31LhA24xU7sb9c1YBLC4sLZ0Cha2SfVLvBPZFXQufHq0M83NRPucXBMPXikfwYQCkzHLDMYxCJR0SiiJEU09kQ_IRIqwrupG95gFf0hTggkPxVtl_giWJVYoGcSjywK0fsdf_W_7z1EkdeTVNnEA&uuid=07a734c1-d6c1-4b13-bccc-ddc234a825e3&authuser=0"
url_helen = "https://pages.cs.wisc.edu/~lizhang/projects/face-parsing/SmithCVPR2013_dataset_resized.zip"

def download_camvid_dataset(url: str = url_camvid):
    """
    """
    urllib.request.urlretrieve(url, filename="archive.zip")
    # os.mkdir("dataset")
    with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
        zip_ref.extractall("dataset")
    
    ##These parts will imply the dataset directory processing and stuff
    # os.remove("archive.zip")


def download_mhp_dataset(url: str = url_mhp_v2):
    """
    """
    urllib.request.urlretrieve(url, filename="LV_MHP.zip")
    # os.mkdir("dataset")
    with zipfile.ZipFile("LV_MHP.zip", 'r') as zip_ref:
        zip_ref.extractall("dataset")
    
    ##These parts will imply the dataset directory processing and stuff
    # os.remove("LV_MHP.zip")

def download_helen_dataset(url: str = url_helen):
    """
    """
    urllib.request.urlretrieve(url, filename="HELEN.zip")
    # os.mkdir("dataset")
    with zipfile.ZipFile("HELEN.zip", 'r') as zip_ref:
        zip_ref.extractall("dataset")

    ##These parts will imply the dataset directory processing and stuff
    # os.remove("HELEN.zip")

def download_data(data_name: str):
    """ Downloads the specified dataset.
    Returns
    -------
    """
    downloader = globals()[f"download_{data_name}_dataset"]
    downloader()