import config
import match_functions
import pandas as pd
import os


def update_mapping_file(df):
    # if mapping file exists, append new matches
    if os.path.exists(config.MAKE_MAPPING_FILEPATH):
        df_old = pd.read_csv(config.MAKE_MAPPING_FILEPATH)
        df = pd.concat([df_old, df])
    # save to mapping file
    df.to_csv(config.MAKE_MAPPING_FILEPATH, index=False)


def get_source_data(source_path):
    df = pd.read_csv(source_path)
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
    std_names = pd.read_csv(config.MEL_PATH)['New Manufacturer'].unique().astype(str)
    std_names = sorted(list(set(std_names)))
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



if __name__ == "__main__":
    # Get source data
    source_names = get_source_data(config.SOURCE_PATH)
    #source_names = source_names[0:1000]
    mapping_dl = get_mel_mapping_dict_list(config.MAKE_MAPPING_FILEPATH)
    preexisting_matches = [d for d in source_names if d in [x['make_source'] for x in mapping_dl]]
    new_source_names = [d for d in source_names if d not in preexisting_matches]
    print('Total makes in source file: {}'.format(str(len(source_names))))
    print('Preexisting matches: {}'.format(str(len(preexisting_matches))))
    print('New makes: {}'.format(str(len(new_source_names))))
    # Map to MEL
    print('Mapping to MEL')
    new_dl = map_to_mel(new_source_names)
    full_dl = mapping_dl + new_dl
    df = pd.DataFrame(full_dl)
    df = df.drop_duplicates(subset=['make_source'])
    update_mapping_file(df)
    """
    df_from_mel = df[df['match_type'].str.lower() != 'no_match']
    print('Found {} matches from MEL'.format(str(len(df_from_mel))))
    
    # Map pending records to mapping file
    if os.path.exists(config.MAKE_MAPPING_FILEPATH):
        print('Mapping to mapping file')
        source_names_pending = sorted(list(df[df['match_type'].str.lower() == 'no_match']['make_source'].unique()))
        df_from_mapping_file = map_to_mapping_file(source_names_pending)
        df = pd.concat([df_from_mel, df_from_mapping_file])
        x = len(df_from_mapping_file[df_from_mapping_file['match_type'].str.lower() != 'no_match'])
        print('Found {} matches from mapping file'.format(str(x)))
    else:
        df = df_from_mel
    # Update mapping file
    update_mapping_file(df) """