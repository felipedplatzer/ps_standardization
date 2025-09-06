import os
from datetime import datetime
import shutil
import pandas as pd
import time
import csv

MODALITY_MAPPER_TEMPERATURE = 0
OPENAI_KEY_FILEPATH = "./../openai_api_key.txt"
TEMPERATURE = 0.0
FILE_FOLDER = 'files'

LETTERS_LIST = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'other']
OVERRIDE_BLANKS = False # Try to find matches for blanks in previous runs

CONCURRENT_WRITE_TIMEOUT_LONG = 60 #Retry 1 min to write to file - it's possible that file is locked by another process (e.,g., if make or model mapping are running in parallel for different cusotmers)
CONCURRENT_WRITE_TIMEOUT_SHORT = 5 #Retry 5 seconds to write to file - it's possible that file is locked by another process (e.,g., if make or model mapping are running in parallel for different cusotmers)


def get_aggregate_customers_filepath():
    aggregate_customers_filepath = f'./files/aggregate_customers_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv'
    return aggregate_customers_filepath

def remove_duplicates(df, unique_cols, tiebreak_cols):
    priority_cols = []
    for i,tiebreak_col in enumerate(tiebreak_cols):
        df["priority_" + str(i)] = df[tiebreak_col].apply(lambda x: 0 if x.strip() == '' else 1)
        priority_cols.append("priority_" + str(i))
    df = df.sort_values(by=priority_cols, ascending=[False for x in priority_cols])
    df = df.drop_duplicates(subset=unique_cols)
    return df

def try_to_write_file(df, filepath, append_to_old, unique_cols, tiebreak_cols):
# if filepath exists
    if os.path.exists(filepath):
        if append_to_old:
            df_old = pd.read_csv(filepath, dtype=str, na_filter=False)
            df = pd.concat([df_old, df])
        # if backups subfolder doesn't exist, create it
        subfolder = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        if not os.path.exists(f'./{subfolder}/backups'):
            os.makedirs(f'./{subfolder}/backups')
        # move filepath to backups folder, add timestamp to filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename_full = filename.replace('.csv', '_' + timestamp + '.csv')
        shutil.move(filepath, f'./{subfolder}/backups/{filename_full}')
    # Remove duplicates
    cols_to_drop = [x for x in df.columns if 'Unnamed:' in x]
    df = df.drop(columns=cols_to_drop)
    df = df.drop_duplicates()
    df = remove_duplicates(df, unique_cols, tiebreak_cols)
    # save df to filepath
    df.to_csv(filepath, index=False, na_rep='')
    return df

def save_new_file(df, filepath, append_to_old = False, timeout = CONCURRENT_WRITE_TIMEOUT_SHORT, unique_cols = None, tiebreak_cols = None):
    time_start = time.time()
    while True:
        try:
            df = try_to_write_file(df, filepath, append_to_old, unique_cols, tiebreak_cols)
            return df
        except:
            if time.time() - time_start > timeout:
                print(f'Failed to write to {filepath} after {timeout} seconds')
                exit()
    

def get_source_rump():
    x = input('Enter the name of the source file. Leave blank for aggregate_customers.py: ')
    if x[-4:] == '.csv':
        return x[:-4]
    else:
        return x


def get_current_filepath(subfolder, source_rump): 
    files = [f for f in os.listdir(subfolder) if os.path.isfile(subfolder + '/' + f) and f == source_rump +'.csv']
    if len(files) > 1: 
        print(f'Error. More than 1 file found in {subfolder} for {source_rump}')
        return None
    elif len(files) == 0:
        print(f'Warning. No files found in {subfolder} for {source_rump}')
        return subfolder + '/' + source_rump + '.csv'
    else:
        return subfolder + '/' + files[0]

def get_first_letter(name):
    if pd.isna(name) or name =='':
        return 'other'
    first_letter = str(name)[0].lower()
    if first_letter in LETTERS_LIST :
        return first_letter
    else:
        return 'other'


def tile_by_first_letter(l):  # returns list of dicts with key = letter and value = subset of l with that letter
    d = {}
    for x in LETTERS_LIST:
        sub_l = [s for s in l if get_first_letter(s) == x]
        if len(sub_l) > 0:
            d[x] = sub_l
    return d

print(f'\n\nOverride blanks set to {str(OVERRIDE_BLANKS)}.\nif override is set to True, the script will try to find matches for blanks in previous runs.\n')


def get_filepaths(source_rump):
    source_filepath  = get_current_filepath('./files/source_data', source_rump)
    serialized_asset_view_filepath = get_current_filepath('./files/serialized_asset_views', source_rump)
    mel_filepath = get_current_filepath(f'./files', 'mel')
    make_mapping_filepath = get_current_filepath(f'./files', 'make_mapping')
    make_override_filepath = get_current_filepath(f'./files', 'make_mapping_manual_override')
    model_mapping_filepath = get_current_filepath(f'./files', 'model_mapping')
    batch_folder = f'./files/batch_files/{source_rump}'
    return {
        'source': source_filepath,
        'serialized_asset_view': serialized_asset_view_filepath,
        'mel': mel_filepath,
        'make_mapping': make_mapping_filepath,
        'make_override': make_override_filepath,
        'model_mapping': model_mapping_filepath,
        'batch_folder': batch_folder
    }




"""
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


def read_clean_csv(filepath):
    cleaned_rows = []
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {k:v for k, v in row.items()}
            cleaned_rows.append(cleaned)
    df = pd.DataFrame(cleaned_rows, dtype=str)
    df = df.fillna('').astype(str)
    return df

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
"""
