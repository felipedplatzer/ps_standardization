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

def get_std_data():
    # Check if mapping file exists
    if os.path.exists(MAKE_MAPPING_FILEPATH):
        df = pd.read_csv(MAKE_MAPPING_FILEPATH)
        # get make_source and make columns as one list
        std_names = list(df['make_source']) + list(df['make'])
    else:

    df = pd.read_csv(MEL_PATH)

    col_list = [
    #'Current ModelId',
    'New ModelId',
    'New Model',    
    'New Lvl 3 Category (Modality)',
    'New Lvl 2 Category',
    'New Lvl 1 Category',
    'New Manufacturer'
    ]

    df = df[col_list].astype(str)
    return df

def get_source_data():

    import pandas as pd

    df = pd.read_csv(SOURCE_PATH)
    df = df[['company_name', 'make_source', 'modality_source', 'model_name_source', 'model_number']].astype(str)
    return df


if __name__ == "__main__":
    # Get MEL data

    df_std = get_std_data()
    # Get source data
    df_source = get_source_data()
    #Standardize manufacturer names
    source_names = sorted(list(df_source['make_source'].unique()))
    std_names = sorted(list(df_std['New Manufacturer'].unique()))
    dl = match_functions.get_all_matches(std_names, source_names)
    df = pd.DataFrame(dl)
    df.to_csv(MANUF_MATCH_OUTPUT_PATH)
    x = df['match_type'].value_counts().reset_index().rename(columns={0: 'n_matches'})
    print(x)
    #Standardize modalities and models
    x =1
    #Update mapping table
