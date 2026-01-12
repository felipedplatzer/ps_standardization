import os
from datetime import datetime
import shutil
import pandas as pd
import time
import csv
import re

MODALITY_MAPPER_TEMPERATURE = 0
OPENAI_KEY_FILEPATH = "./openai_api_key.txt"
TEMPERATURE = 0.0
FILE_FOLDER = 'files'
MAX_WORKERS = 10 # NUMBER OF threads parallel-processing a set of customers

LETTERS_LIST = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'other']
REMAP_UNMATCHED = False # Try to re-map models where model_match_type = 'no_match'

CONCURRENT_WRITE_TIMEOUT_LONG = 60 #Retry 1 min to write to file - it's possible that file is locked by another process (e.,g., if make or model mapping are running in parallel for different cusotmers)
CONCURRENT_WRITE_TIMEOUT_SHORT = 5 #Retry 5 seconds to write to file - it's possible that file is locked by another process (e.,g., if make or model mapping are running in parallel for different cusotmers)

# MEL Column Mappings - Map output field names to MEL file column names
# Update these values to match your MEL file's actual column names
MEL_COLUMNS = {
    'mel_id': 'MODEL_ID',                    # MEL ID column in MEL file
    'make_target': 'MANUFACTURER_NAME',      # Manufacturer column in MEL file  
    'model_name_target': 'MODEL_NAME',       # Model name column in MEL file
    'l1_modality_target': 'L1_TAXONOMY',     # L1 modality (most general, e.g., continent)
    'l2_modality_target': 'L2_TAXONOMY',     # L2 modality (mid-level, e.g., country)
    'l3_modality_target': 'L3_TAXONOMY',      # L3 modality (most specific, e.g., city)
    'manufacturer_aliases': 'MANUFACTURER_ALIASES'
}

# Source file column mappings for modality fields
# Update these values to match your source file's actual column names
SOURCE_COLUMNS = {
    'l1_modality_source': 'l1_modality_source',  # L1 modality column in source file
    'l2_modality_source': 'l2_modality_source',  # L2 modality column in source file
    'l3_modality_source': 'l3_modality_source'   # L3 modality column in source file
}


def normalize_name(name):
    """
    Normalize a name by:
    1. Converting to uppercase
    2. Removing special characters (keeping only alphanumeric and spaces)
    3. Trimming whitespace
    """
    if pd.isna(name) or name is None:
        return ''
    name = str(name).upper()
    # Remove special characters, keep only alphanumeric and spaces
    name = re.sub(r'[^A-Z0-9\s]', '', name)
    # Collapse multiple spaces into one and trim
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def get_customer_summary_filepath():
    return f'./files/customer_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx'

def get_customer_joined_filepath():
    return f'./files/serialized_asset_view_all_customers_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx'

def remove_duplicates(df, unique_cols, tiebreak_cols):
    priority_cols = []
    for i,tiebreak_col in enumerate(tiebreak_cols):
        df["priority_" + str(i)] = df[tiebreak_col].apply(lambda x: 0 if x.strip() == '' else 1)
        priority_cols.append("priority_" + str(i))
    df = df.sort_values(by=priority_cols, ascending=[False for x in priority_cols])
    df = df.drop_duplicates(subset=unique_cols)
    df = df.drop(columns=priority_cols)
    return df

def try_to_write_file(df, filepath, append_to_old, unique_cols, tiebreak_cols, print_stats = False):
# if filepath exists
    if os.path.exists(filepath):
        if append_to_old:
            df_old = pd.read_excel(filepath, dtype=str, na_filter=False)
            df_old = df_old.apply(lambda x: x.str.strip())
            if print_stats:
                print(f'N rows in old file: {str(len(df_old))}')
                print(f'N new rows: {str(len(df))}')
            df = pd.concat([df_old, df])
        # if backups subfolder doesn't exist, create it
        subfolder = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        if not os.path.exists(f'./{subfolder}/backups'):
            os.makedirs(f'./{subfolder}/backups')
        # move filepath to backups folder, add timestamp to filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename_full = filename.replace('.xlsx', '_' + timestamp + '.xlsx')
        shutil.move(filepath, f'./{subfolder}/backups/{filename_full}')
    # Remove duplicates
    cols_to_drop = [x for x in df.columns if 'Unnamed:' in x]
    df = df.drop(columns=cols_to_drop)
    df = df.drop_duplicates()
    df = remove_duplicates(df, unique_cols, tiebreak_cols)
    # save df to filepath
    if print_stats:
        print(f'N rows in new file: {str(len(df))}')
    df.to_excel(filepath, index=False, na_rep='')
    return df

def save_new_file(df, filepath, append_to_old = False, timeout = CONCURRENT_WRITE_TIMEOUT_SHORT, unique_cols = None, tiebreak_cols = None, print_stats = False):
    time_start = time.time()
    while True:
        try:
            df = try_to_write_file(df, filepath, append_to_old, unique_cols, tiebreak_cols, print_stats)
            return df
        except:
            if time.time() - time_start > timeout:
                print(f'Failed to write to {filepath} after {timeout} seconds')
                exit()
    

def get_source_rump():
    """
    Get source file name(s) from user input.
    
    Returns:
        list: List of source file names (without extension)
    
    Supported inputs:
        - Single filename: "customer.xlsx" or "customer"
        - Multiple filenames separated by commas: "customer1.xlsx, customer2.xlsx"
        - Blank (empty string or just pressing Enter): Process all .xlsx files in source_data folder
    """
    x = input('Enter source file name(s) (comma-separated for multiple, blank for all files in source_data): ')
    x = x.strip()
    
    # If blank, get all files in source_data folder
    if x == '':
        source_folder = './files/source_data'
        if os.path.exists(source_folder):
            all_files = [f[:-5] for f in os.listdir(source_folder) 
                        if f.endswith('.xlsx') and os.path.isfile(os.path.join(source_folder, f))]
            if len(all_files) == 0:
                print('No .xlsx files found in source_data folder')
                return []
            print(f'Found {len(all_files)} files: {", ".join(all_files)}')
            return sorted(all_files)
        else:
            print(f'Source folder {source_folder} does not exist')
            return []
    
    # Split by comma and process each filename
    filenames = [f.strip() for f in x.split(',')]
    result = []
    source_folder = './files/source_data'
    
    for filename in filenames:
        if filename.endswith('.xlsx'):
            file_rump = filename[:-5]
        else:
            file_rump = filename
        
        # Check if file exists
        file_path = os.path.join(source_folder, file_rump + '.xlsx')
        if os.path.exists(file_path):
            result.append(file_rump)
        else:
            print(f'Warning: File not found: {file_path}')
    
    if len(result) == 0:
        print('ERROR: No matching files found in source_data folder. Stopping.')
        return []
    
    if len(result) < len(filenames):
        print(f'Found {len(result)} of {len(filenames)} specified files')
    
    return result


def get_source_rumps():
    """Alias for get_source_rump that returns list of source file names."""
    return get_source_rump()


def get_current_filepath(subfolder, source_rump, extension = '.xlsx'): 
    files = [f for f in os.listdir(subfolder) if os.path.isfile(subfolder + '/' + f) and f == source_rump + extension]
    if len(files) > 1: 
        print(f'Error. More than 1 file found in {subfolder} for {source_rump}')
        return None
    elif len(files) == 0:
        print(f'Warning. No files found in {subfolder} for {source_rump}')
        return subfolder + '/' + source_rump + extension
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

if REMAP_UNMATCHED:
    print(f'\n\nRemap unmatched set to True.\nThe script will try to re-map models where model_match_type = "no_match".\n')
else:
    print(f'\n\nRemap unmatched set to False.\nThe script will skip previously unmatched models.\n')


def get_filepaths(source_rump):
    source_filepath  = get_current_filepath('./files/source_data', source_rump)
    serialized_asset_view_filepath = get_current_filepath('./files/serialized_asset_views', source_rump)
    mel_filepath = get_current_filepath(f'./files', 'mel', '.xlsx')
    make_mapping_filepath = get_current_filepath(f'./files', 'make_mapping')
    make_override_filepath = get_current_filepath(f'./files', 'make_mapping_manual_override', '.xlsx')
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