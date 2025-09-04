import config
import match_functions
import pandas as pd
import os
from llm_make_mapper import llm_make_mapper

def update_mapping_file(df):
    # if mapping file exists, append new matches
    if os.path.exists(config.MAKE_MAPPING_FILEPATH):
        df_old = pd.read_csv(config.MAKE_MAPPING_FILEPATH)
        df = pd.concat([df_old, df])
    # save to mapping file
    df.to_csv(config.MAKE_MAPPING_FILEPATH, index=False)


def get_source_data(source_path):
    df = pd.read_csv(source_path)
    df['make_source'] = df['make_source'].fillna('')
    df = list(df['make_source'].astype(str).unique())
    return sorted(df)


def get_mel_mapping_dict_list(mel_path):
    if os.path.exists(config.MAKE_MAPPING_FILEPATH):
        df = pd.read_csv(config.MAKE_MAPPING_FILEPATH)
        dl = df.to_dict(orient='records')
        return dl
    else:
        return []

def map_to_mel(source_names):
    s = pd.read_csv(config.MEL_FILEPATH)['New Manufacturer'].fillna('').astype(str)
    std_names = list(s.unique())
    std_names = sorted(std_names)
    dl = match_functions.get_all_matches(std_names, source_names)
    return dl


def map_to_mapping_file(source_names):
    # Get list of standardized names
    map_df = pd.read_csv(config.MAKE_MAPPING_FILEPATH)
    map_df = map_df[map_df['confirmed'] == True] # remove not confirmed matches
    map_df = map_df[['make_source', 'make_target']] # take only relevant columns
    std_names = list(map_df['make_source'].astype(str)) + list(map_df['make_target'].astype(str)) # get make_source and make columns as one list
    std_names = sorted(list(set(std_names))) # remove duplicates and sort
    # Get matches for pending records
    dl = match_functions.get_all_matches(std_names, source_names)
    df_2 = pd.DataFrame(dl)
    # Swap make by make from mel
    map_df = map_df.rename(columns={'make_source': 'make_mapping_file'})
    df_2 = df_2.rename(columns={'make_target': 'make_mapping_file'})
    df_3 = pd.merge(df_2, map_df, on='make_mapping_file', how='left')
    df_3 = df_3.drop(columns=['make_mapping_file'])
    return df_3

def remove_preexisting_matches(source_names, make_mapping_filepath):
    if os.path.exists(make_mapping_filepath):
        make_mapping_df = pd.read_csv(make_mapping_filepath)
        # REMOVE umapped from set of existing makesonsideratoin
        if config.OVERRIDE_BLANKS:
            make_mapping_df = make_mapping_df[make_mapping_df['make_target'].notna()]
        preexisting_make_source_names = list(make_mapping_df['make_source'].unique())
        new_source_names = [d for d in source_names if d not in preexisting_make_source_names]
    else:
        new_source_names = source_names
    return new_source_names

if __name__ == "__main__":
    # Get source data
    source_names = get_source_data(config.SOURCE_FILEPATH)
    #source_names = source_names[0:200]
    #source_names = source_names[0:1000]
    new_source_names = remove_preexisting_matches(source_names, config.MAKE_MAPPING_FILEPATH)
    n_preexisting_matches = len(source_names) - len(new_source_names)
    print('Total makes in source file: {}'.format(str(len(source_names))))
    print('Preexisting matches: {}'.format(str(n_preexisting_matches)))
    print('New makes: {}'.format(str(len(new_source_names))))
    # Map to MEL
    print('Mapping to MEL')
    dl = map_to_mel(new_source_names)
    if len(dl) > 0:
        df = pd.DataFrame(dl)
        df = df.drop_duplicates(subset=['make_source'])
        config.save_new_file(df, config.MAKE_MAPPING_FILEPATH, append_to_old=True, timeout=config.CONCURRENT_WRITE_TIMEOUT_LONG)
