from modelling2.util import ascii_to_gesture

SEED_CONSTANT = 23
CLASSES_LIST = ascii_to_gesture.values()
NUM_CLASSES = len(CLASSES_LIST)

IMG_HEIGHT = 60
IMG_WIDTH = 60
MAX_IMGS_PER_CLASS = 1000
DATA_DIR = "video_data_D"

NUM_EPOCHS = 60
BATCH_SIZE = 16
