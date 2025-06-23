SEMARCHY_PATH = "C:/Users/FelipePlatzer/Documents/Work_2025/PartsSource/Development/Uptime data ingestion - old and tests/old and tests/semarchy_export_2025-06-06.csv"
MEL_PATH = "C:/Users/FelipePlatzer/Documents/Work_2025/PartsSource/Development/Uptime data ingestion/assets/mapping_tables/mel.csv"
SOURCE_PATH = "C:/Users/FelipePlatzer/Documents/Work_2025/PartsSource/Development/Uptime data ingestion/assets/outputs/silver_assets.csv"

MAKE_MAPPING_FILEPATH = "C:/Users/FelipePlatzer/Documents/Work_2025/PartsSource/Development/Uptime data ingestion/stand_attempt_2025-06-18/make_mapping.csv"
MODEL_MODALITY_MAPPING_FILEPATH = "C:/Users/FelipePlatzer/Documents/Work_2025/PartsSource/Development/Uptime data ingestion/stand_attempt_2025-06-18/model_modality_mapping.csv"

import match_functions
import pandas as pd
import os
import re
import llm_model_mapper



def update_mapping_file(df):
    # if mapping file exists, append new matches
    if os.path.exists(MAKE_MAPPING_FILEPATH):
        df_old = pd.read_csv(MAKE_MAPPING_FILEPATH)
        df = pd.concat([df_old, df])
    # save to mapping file
    df.to_csv(MAKE_MAPPING_FILEPATH, index=False)


def get_source_data():
    df = pd.read_csv(SOURCE_PATH)
    df = df[['make_source', 'model_name_source', 'model_number', 'modality_source']].fillna('').astype(str)
    return df

def get_target_data():
    df = pd.read_csv(MEL_PATH)
    df = df[['New ModelId', 'New Model', 'New Lvl 2 Category', 'New Manufacturer']]
    df['New ModelId'] = pd.to_numeric(df['New ModelId'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['New ModelId'])
    df = df.fillna('').astype(str)
    df = df.rename(columns={'New ModelId': 'mel_id', 'New Manufacturer': 'make_target', 'New Model': 'model_name_target', 'New Lvl 2 Category': 'modality_target'})
    return df


def standardize_make(source_df):
    # Get standardized manufacturer name
    std_df = pd.read_csv(MAKE_MAPPING_FILEPATH)
    df = pd.merge(source_df, std_df, on='make_source', how='inner') # Remove unmapped manufacturers
    df = df.drop(columns=['make_source'])
    return df


def match_model(source_model, target_model_list):
    # Find exact match
    for target_model in target_model_list:
        if source_model.lower() == target_model.lower():
            return target_model, 'exact', True # true means confirmed. exact matches don't need confirmation
    # Find match skipping special characters and spaces
    target_model_list = [re.sub(r'[^a-zA-Z0-9]', '', x.lower()) for x in target_model_list]
    source_model = re.sub(r'[^a-zA-Z0-9]', '', source_model.lower())
    for target_model in target_model_list:
        if source_model == target_model:
            return target_model, 'skip_special_chars', False # false means not confirmed. partial matches  need confirmation
    # If no match, return empty string
    return '', 'no_match', False



def map_one_make(source_df, target_df, make_source):
    # Init
    dl = []
    # Get source and target data for this manufacturer
    source_df_make = source_df[source_df['make_target'] == make_source]
    target_df_make = target_df[target_df['make_target'] == make_source]
    target_model_list = list(target_df_make['model_name_target'].unique())
    target_model_list = [x for x in target_model_list if x.strip() != '']
    source_model_list = list(source_df_make['model_name_source'].unique())
    # Deterministic mapping
    # Loop through source models
    for source_model in source_model_list:
        # Find match for each target model
        target_model, match_type, confirmed = match_model(source_model, target_model_list)
        match_dict = {'model_name_source': source_model, 'model_name_target': target_model, 'match_type': match_type, 'confirmed': confirmed}
        dl.append(match_dict)
    # LLM mapping
    # Batch all remaining, unmatched models
    unmatched_dl = [x for x in dl if x['match_type'] == 'no_match']
    deterministic_matched_dl = [x for x in dl if x['match_type'] != 'no_match']
    if len(unmatched_dl) > 0 and len(target_model_list) > 0:
        # Use LLM to match unmatched models to target_model_list
        llm_dl = llm_model_mapper.main(target_model_list, unmatched_dl)
        # Post-process unmapped models
        llm_matched_dl = [x for x in llm_dl if x['model_name_target'] != 'NO_MATCH']
        for x in llm_matched_dl: 
            x['match_type'] = 'llm'
            x['confirmed'] = False
        unmatched_dl = [x for x in llm_dl if x['model_name_target'] == 'NO_MATCH']
        for x in unmatched_dl: 
            x['match_type'] = 'no_match'
            x['model_name_target'] = ''
            x['confirmed'] = False
    else:
        llm_matched_dl = []
    # Join deterministic and LLM matches and no matches
    dl = deterministic_matched_dl + llm_matched_dl + unmatched_dl
    # Add modality - PENDING!!
    # convert to dataframe
    df = pd.DataFrame(dl)
    return df


def get_mel_id(df):

    # Read MEL file 
    mel_df = pd.read_csv(MEL_PATH)
    mel_df = mel_df[['New ModelId', 'New Model', 'New Lvl 2 Category', 'New Manufacturer']]
    mel_df = mel_df.rename(columns={'New ModelId': 'mel_id', 'New Manufacturer': 'make_target', 'New Model': 'model_name_target', 'New Lvl 2 Category': 'modality_target'})
    # Join based on model_name_target and make_target
    df = pd.merge(df, mel_df, on=['model_name_target', 'make_target'], how='left')
    return df


if __name__ == "__main__":
    # Get source data
    source_df = get_source_data()
    # Get standardized manufacturer name (removes unmapped manufacturers)
    source_df = standardize_make(source_df)
    # Get target data
    target_df = get_target_data()
    # Loop through mapped manufactuers
    df_list = []
    x = list(source_df['make_target'].unique())[0:3]
    for i, make_target in enumerate(x):
        print(f"Processing manufacturer {i} of {len(source_df['make_target'].unique())}")
        # Map model
        make_df = map_one_make(source_df, target_df, make_target)
        # Add make 
        make_df['make_target'] = make_target
        # Add to list
        df_list.append(make_df)        
    # Join df's
    df = pd.concat(df_list)
    # Get MEL ID
    df = get_mel_id(df)
    # save to csv
    df.to_csv(MODEL_MODALITY_MAPPING_FILEPATH)
