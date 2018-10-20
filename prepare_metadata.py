import os

import neptune
from sklearn.externals import joblib
import pandas as pd
from tqdm import tqdm

from common_blocks.utils import misc, masks

CTX = neptune.Context()
LOGGER = misc.init_logger()

DEV_MODE = True

#    ______   ______   .__1   __.  _______  __    _______      _______.
#   /      | /  __  \  |  \ |  | |   ____||  |  /  _____|    /       |
#  |  ,----'|  |  |  | |   \|  | |  |__   |  | |  |  __     |   (----`
#  |  |     |  |  |  | |  . `  | |   __|  |  | |  | |_ |     \   \
#  |  `----.|  `--'  | |  |\   | |  |     |  | |  |__| | .----)   |
#   \______| \______/  |__| \__| |__|     |__|  \______| |_______/
#

ORIGINAL_SIZE = (768, 768)
EXCLUDED_FILENAMES = ['6384c3e78.jpg', ]

if CTX.params.__class__.__name__ == 'OfflineContextParams':
    PARAMS = misc.read_yaml().parameters
else:
    PARAMS = CTX.params


#   __________   ___  _______   ______  __    __  .___________. __    ______   .__   __.
#  |   ____\  \ /  / |   ____| /      ||  |  |  | |           ||  |  /  __  \  |  \ |  |
#  |  |__   \  V  /  |  |__   |  ,----'|  |  |  | `---|  |----`|  | |  |  |  | |   \|  |
#  |   __|   >   <   |   __|  |  |     |  |  |  |     |  |     |  | |  |  |  | |  . `  |
#  |  |____ /  .  \  |  |____ |  `----.|  `--'  |     |  |     |  | |  `--'  | |  |\   |
#  |_______/__/ \__\ |_______| \______| \______/      |__|     |__|  \______/  |__| \__|
#

def prepare_masks():
    LOGGER.info('overlaying masks')
    overlay_masks(annotation_file_name=PARAMS.annotation_file, target_dir=PARAMS.masks_overlayed_dir)


def prepare_metadata():
    LOGGER.info('creating metadata')
    meta = generate_metadata(train_images_dir=PARAMS.train_images_dir,
                             masks_overlayed_dir=PARAMS.masks_overlayed_dir,
                             test_images_dir=PARAMS.test_images_dir,
                             annotation_file_name=PARAMS.annotation_file
                             )
    meta.to_csv(os.path.join(PARAMS.metadata_filepath), index=None)


#   __    __  .___________. __   __          _______.
#  |  |  |  | |           ||  | |  |        /       |
#  |  |  |  | `---|  |----`|  | |  |       |   (----`
#  |  |  |  |     |  |     |  | |  |        \   \
#  |  `--'  |     |  |     |  | |  `----.----)   |
#   \______/      |__|     |__| |_______|_______/
#

def overlay_masks(annotation_file_name, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    annotations = pd.read_csv(annotation_file_name)

    for file_name, image_annotation in tqdm(annotations.groupby("ImageId")):
        if file_name not in EXCLUDED_FILENAMES:
            target_file_name = os.path.join(target_dir, file_name.split('.')[0])
            mask = masks.get_overlayed_mask(image_annotation, ORIGINAL_SIZE)
            if mask.sum() == 0:
                mask = ORIGINAL_SIZE
            joblib.dump(mask, target_file_name)


def generate_metadata(train_images_dir, masks_overlayed_dir, test_images_dir, annotation_file_name):
    metadata = {}
    annotations = pd.read_csv(annotation_file_name)
    for filename in tqdm(os.listdir(train_images_dir)):
        image_filepath = os.path.join(train_images_dir, filename)
        mask_filepath = os.path.join(masks_overlayed_dir, filename.split('.')[0])
        image_id = filename.split('.')[0]
        number_of_ships = get_number_of_ships(annotations.query('ImageId == @filename'))
        is_not_empty = int(number_of_ships != 0)

        metadata.setdefault('file_path_image', []).append(image_filepath)
        metadata.setdefault('file_path_mask', []).append(mask_filepath)
        metadata.setdefault('is_train', []).append(1)
        metadata.setdefault('id', []).append(image_id)
        metadata.setdefault('number_of_ships', []).append(number_of_ships)
        metadata.setdefault('is_not_empty', []).append(is_not_empty)

    for filename in tqdm(os.listdir(test_images_dir)):
        image_filepath = os.path.join(test_images_dir, filename)
        image_id = filename.split('.')[0]

        metadata.setdefault('file_path_image', []).append(image_filepath)
        metadata.setdefault('file_path_mask', []).append(None)
        metadata.setdefault('is_train', []).append(0)
        metadata.setdefault('id', []).append(image_id)
        metadata.setdefault('number_of_ships', []).append(None)
        metadata.setdefault('is_not_empty', []).append(None)

    return pd.DataFrame(metadata)


def get_number_of_ships(image_annotations):
    if image_annotations['EncodedPixels'].any():
        return len(image_annotations)
    else:
        return 0


#  .___  ___.      ___       __  .__   __.
#  |   \/   |     /   \     |  | |  \ |  |
#  |  \  /  |    /  ^  \    |  | |   \|  |
#  |  |\/|  |   /  /_\  \   |  | |  . `  |
#  |  |  |  |  /  _____  \  |  | |  |\   |
#  |__|  |__| /__/     \__\ |__| |__| \__|
#

if __name__ == "__main__":
    prepare_masks()
    prepare_metadata()
