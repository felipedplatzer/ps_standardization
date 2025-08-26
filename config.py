MEL_PATH = "./mel.csv"
SUBFOLDER = input('Enter the subfolder name. Leave blank for aggregate_customers.py: ')
import os
from datetime import datetime
import shutil

MODALITY_MAPPER_TEMPERATURE = 0
OPENAI_KEY_FILEPATH = "./../openai_api_key.txt"

GE_MAKES = { #note: all keys have to be lowercase, values are the correct name
    'ge': 'GE Healthcare',
    'ge medical systems': 'GE Healthcare',
    'ge health care': 'GE Healthcare',
    'ge analytical instruments': 'GE Healthcare',
    'ge healthcare': 'GE Healthcare',
    'ge hc': 'GE Healthcare',
    'gehc': 'GE Healthcare',
    'ge healthcare usa': 'GE Healthcare',
    'ge medical systems': 'GE Healthcare',
    'ge medical critikon; inc.': 'GE Healthcare',
    'ge healthcare technologies': 'GE Healthcare',
    'ge healthcare usa (imaging)': 'GE Healthcare',
    'ge oec medical systems': 'GE Healthcare'
}

def make_override(make_raw):
    make_raw = make_raw.lower().strip()
    if make_raw in GE_MAKES:
        return GE_MAKES[make_raw]
    elif 'siemens' in make_raw:
        return 'Siemens'
    elif 'philips' in make_raw:
        return 'Philips'
    else:
        return make_raw

def get_source_path(subfolder=SUBFOLDER):
    source_path = f'./{subfolder}/source.csv'
    return source_path


def get_mel_path(subfolder=SUBFOLDER):
    mel_path = f'./{subfolder}/mel.csv'
    return mel_path


def get_make_mapping_filepath(subfolder=SUBFOLDER):
    make_mapping_filepath = f'./{subfolder}/make_mapping.csv'
    return make_mapping_filepath


def get_model_mapping_filepath(subfolder=SUBFOLDER):
    model_mapping_filepath = f'./{subfolder}/model_mapping.csv'
    return model_mapping_filepath



def get_serialized_asset_view_filepath(subfolder=SUBFOLDER):
    serialized_asset_view_filepath = f'./{subfolder}/serialized_asset_view.csv'
    return serialized_asset_view_filepath


def get_aggregate_customers_filepath():
    aggregate_customers_filepath = f'./aggregate_customers_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv'
    return aggregate_customers_filepath

def get_batch_files_path(subfolder=SUBFOLDER):
    if not os.path.exists(f'./{subfolder}/batch_files'):
        os.makedirs(f'./{subfolder}/batch_files')
    batch_files_path = f'./{subfolder}/batch_files'
    return batch_files_path




def save_new_file(df, filepath, subfolder=SUBFOLDER):
    # if filepath exists
    if os.path.exists(filepath):
        # if backups subfolder doesn't exist, create it
        if not os.path.exists(f'./{subfolder}/backups'):
            os.makedirs(f'./{subfolder}/backups')
        # move filepath to backups folder, add timestamp to filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # remove .csv from filename
        filename = os.path.basename(filepath).replace('.csv', '')
        shutil.move(filepath, f'./{subfolder}/backups/{filename}_{timestamp}.csv')
    # save df to filepath
    df.to_csv(filepath, index=False)



SOURCE_PATH = get_source_path()
MEL_PATH = get_mel_path()
MAKE_MAPPING_FILEPATH = get_make_mapping_filepath()
MODEL_MAPPING_FILEPATH = get_model_mapping_filepath()
SERIALIZED_ASSET_VIEW_FILEPATH = get_serialized_asset_view_filepath()
BATCH_FILES_PATH = get_batch_files_path()





