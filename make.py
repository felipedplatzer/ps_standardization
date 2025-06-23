SEMARCHY_PATH = "C:/Users/FelipePlatzer/Documents/Work_2025/PartsSource/Development/Uptime data ingestion - old and tests/old and tests/semarchy_export_2025-06-06.csv"
MEL_PATH = "C:/Users/FelipePlatzer/Documents/Work_2025/PartsSource/Development/Uptime data ingestion/assets/mapping_tables/mel.csv"
SOURCE_PATH = "C:/Users/FelipePlatzer/Documents/Work_2025/PartsSource/Development/Uptime data ingestion/assets/outputs/silver_assets.csv"

MAKE_MAPPING_FILEPATH = "C:/Users/FelipePlatzer/Documents/Work_2025/PartsSource/Development/Uptime data ingestion/stand_attempt_2025-06-18/make_mapping.csv"

import match_functions
import pandas as pd
import os

# semarchy
def get_semarchy_data():

    import pandas as pd

    df = pd.read_csv(SEMARCHY_PATH)

    print(df.head())

    col_list = [
        'GOLDEN_MODEL_ID',
        'GOLDEN_MODEL_NAME',
        'GOLDEN_MANUFACTURER_NAME',
        'ORIG_MODEL_SOURCE',
        'ORIG_MODEL_ID',
        'ORIG_MODEL_NAME',
        'ORIG_MANUFACTURER_NAME',
        'ORIG_STRUC_PATH',
        'TAXONOMY_PATH'
    ]

    df = df[col_list]
    return df




def update_mapping_file(df):
    # if mapping file exists, append new matches
    if os.path.exists(MAKE_MAPPING_FILEPATH):
        df_old = pd.read_csv(MAKE_MAPPING_FILEPATH)
        df = pd.concat([df_old, df])
    # save to mapping file
    df.to_csv(MAKE_MAPPING_FILEPATH, index=False)


def get_source_data():
    df = pd.read_csv(SOURCE_PATH)
    df = list(df['make_source'].astype(str).unique())
    return sorted(df)


def map_to_mel(source_names):
    std_names = pd.read_csv(MEL_PATH)['New Manufacturer'].unique().astype(str)
    std_names = sorted(list(set(std_names)))
    dl = match_functions.get_all_matches(std_names, source_names)
    df = pd.DataFrame(dl)
    return df


def map_to_mapping_file(source_names):
    # Get list of standardized names
    map_df = pd.read_csv(MAKE_MAPPING_FILEPATH)
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
    source_names = get_source_data()
    #source_names = source_names[0:1000]
    # Map to MEL
    print('Mapping to MEL')
    df = map_to_mel(source_names)
    df_from_mel = df[df['match_type'] != 'no_match']
    print('Found {} matches from MEL'.format(str(len(df_from_mel))))
    # Map pending records to mapping file
    if os.path.exists(MAKE_MAPPING_FILEPATH):
        print('Mapping to mapping file')
        source_names_pending = sorted(list(df[df['match_type'] == 'no_match']['make_source'].unique()))
        df_from_mapping_file = map_to_mapping_file(source_names_pending)
        df = pd.concat([df_from_mel, df_from_mapping_file])
        x = len(df_from_mapping_file[df_from_mapping_file['match_type'] != 'no_match'])
        print('Found {} matches from mapping file'.format(str(x)))
    else:
        df = df_from_mel
    # Update mapping file
    update_mapping_file(df)
    df.to_csv(MAKE_MAPPING_FILEPATH)
 