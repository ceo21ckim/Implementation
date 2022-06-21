import os

BASE_DIR = os.path.dirname(__file__)

ZIP_DIR = os.path.join(BASE_DIR, 'data_gzip')
PICKLE_DIR = os.path.join(BASE_DIR, 'pickle_dataset')
TXT_DIR = os.path.join(BASE_DIR, 'txt_dataset')

SPM_PATH = os.path.join(BASE_DIR, 'vocaburary')
EMD_PATH = os.path.join(BASE_DIR, 'word_embedding')

DATA_NAMES = ['beauty', 'electronics', 'health', 'sport_outdoors']
