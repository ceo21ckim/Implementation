import os

BASE_DIR = os.path.dirname(__file__)

ZIP_DIR = os.path.join(BASE_DIR, 'data_gzip')
PICKLE_DIR = os.path.join(BASE_DIR, 'pickle_dataset')
TXT_DIR = os.path.join(BASE_DIR, 'txt_dataset')

SPM_PATH = os.path.join(BASE_DIR, 'vocaburary')
EMD_PATH = os.path.join(BASE_DIR, 'word_embedding')
IMG_PATH = os.path.join(BASE_DIR, 'image')


DATA_NAMES = ['beauty', 'electronics', 'health', 'sport_outdoors']


if not os.path.exists(ZIP_DIR):
    os.makedirs(ZIP_DIR)

if not os.path.exists(PICKLE_DIR):
    os.makedirs(PICKLE_DIR)

if not os.path.exists(TXT_DIR):
    os.makedirs(TXT_DIR)

if not os.path.exists(SPM_PATH):
    os.makedirs(SPM_PATH)

if not os.path.exists(EMD_PATH):
    os.makedirs(EMD_PATH)

if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)
