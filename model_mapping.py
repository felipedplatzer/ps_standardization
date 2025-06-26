SEMARCHY_PATH = "C:/Users/FelipePlatzer/Documents/Work_2025/PartsSource/Development/Uptime data ingestion - old and tests/old and tests/semarchy_export_2025-06-06.csv"
MEL_PATH = "C:/Users/FelipePlatzer/Documents/Work_2025/PartsSource/Development/Uptime data ingestion/assets/mapping_tables/mel.csv"
# SOURCE_PATH = "C:/Users/FelipePlatzer/Documents/Work_2025/PartsSource/Development/Uptime data ingestion/assets/outputs/silver_assets.csv"
SOURCE_PATH = "C:/Users/FelipePlatzer/Documents/Work_2025/PartsSource/Development/Uptime data ingestion/stand_attempt_2025-06-18/silver_assets_SUMMA.csv"

MAKE_MAPPING_FILEPATH = "C:/Users/FelipePlatzer/Documents/Work_2025/PartsSource/Development/Uptime data ingestion/stand_attempt_2025-06-18/make_mapping.csv"
MODEL_MODALITY_MAPPING_FILEPATH = "C:/Users/FelipePlatzer/Documents/Work_2025/PartsSource/Development/Uptime data ingestion/stand_attempt_2025-06-18/model_mapping.csv"
SERIALIZED_ASSET_VIEW_FILEPATH = "C:/Users/FelipePlatzer/Documents/Work_2025/PartsSource/Development/Uptime data ingestion/stand_attempt_2025-06-18/serialized_asset_view.csv"

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
    # Get MEL data
    df = pd.read_csv(MEL_PATH)
    df = df[['New ModelId', 'New Model', 'New Manufacturer', 'New Lvl 2 Category']]
    df['New ModelId'] = pd.to_numeric(df['New ModelId'], errors='coerce').astype('Int64')
    df = df.rename(columns={'New ModelId': 'mel_id', 'New Manufacturer': 'make_target', 'New Model': 'model_name_target', 'New Lvl 2 Category': 'modality_target'  }) # 'New Lvl 2 Category': 'modality_target' ###
    # Get crosswalk data
    if os.path.exists(MODEL_MODALITY_MAPPING_FILEPATH):
        df_2 = pd.read_csv(MODEL_MODALITY_MAPPING_FILEPATH)
        df_2['mel_id'] = pd.to_numeric(df_2['mel_id'], errors='coerce').astype('Int64')
        df_2 = df_2[['mel_id','make_target', 'model_name_source', 'confirmed']]
        df_2 = df_2.rename(columns={'model_name_source': 'model_name_target'}) # source name of crosswalk file becomes target name for new data
        # Get only confirmed records
        df_2 = df_2[df_2['confirmed'] == True]
        df_2 = df_2.drop(columns=['confirmed'])
        df = pd.concat([df, df_2])
    # post process
    df = df[df['make_target'].notna() & df['model_name_target'].notna()]
    df = df.drop_duplicates()
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
            return target_model, 'exact'
    # Find match skipping special characters and spaces
    target_model_list = [re.sub(r'[^a-zA-Z0-9]', '', x.lower()) for x in target_model_list]
    source_model = re.sub(r'[^a-zA-Z0-9]', '', source_model.lower())
    for target_model in target_model_list:
        if source_model == target_model:
            return target_model, 'skip_special_chars'
    # If no match, return empty string
    return '', 'no_match'


def map_one_make(this_make_source_df, this_make_target_df):
    # Init
    dl = []
    # Get source and target data for this manufacturer
    source_model_list = list(this_make_source_df['model_name_source'].unique())
    source_model_list = [x for x in source_model_list if x.strip() != '']
    target_model_list = list(this_make_target_df['model_name_target'].unique())
    target_model_list = [x for x in target_model_list if x.strip() != '']
    # Deterministic mapping
    # Loop through source models
    for source_model in source_model_list:
        # Find match for each target model
        target_model, match_type = match_model(source_model, target_model_list)
        match_dict = {'model_name_source': source_model, 'model_name_target': target_model, 'match_type': match_type}
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
        unmatched_dl = [x for x in llm_dl if x['model_name_target'] == 'NO_MATCH']
        for x in unmatched_dl: 
            x['match_type'] = 'no_match'
            x['model_name_target'] = ''
    else:
        llm_matched_dl = []
    # Join deterministic and LLM matches and no matches
    dl = deterministic_matched_dl + llm_matched_dl + unmatched_dl
    # convert to dataframe
    df = pd.DataFrame(dl)
    return df

def postprocess_make_df(make_df, this_make_target_df, make_target):
    # Remove duplicates
    this_make_target_df = this_make_target_df.drop_duplicates(subset=['model_name_target'])
    # Backfill MEL ID
    make_df = pd.merge(make_df, this_make_target_df, on='model_name_target', how='left')
    # Add make 
    make_df['make_target'] = make_target
    # add match type
    make_df['confirmed'] = make_df['match_type'].apply(lambda x: True if x == 'exact' else False)
    return make_df


def join_batch_files():
    # Get all batch files
    batch_files = [os.path.join('./batch_files', file) for file in os.listdir('./batch_files')]
    # Join all batch files
    df = pd.concat([pd.read_csv(file) for file in batch_files])
    return df

def create_serialized_asset_view(df):
    # Read source data
    source_df = pd.read_csv(SOURCE_PATH)
    source_df = source_df[['company_name','asset_sys_id','make_source', 'model_name_source', 'model_number', 'modality_source']].fillna('').astype(str)
    source_df = source_df.drop_duplicates(subset=['asset_sys_id'])
    # Standardize make
    make_mapping_df = pd.read_csv(MAKE_MAPPING_FILEPATH)
    output_df = pd.merge(source_df, make_mapping_df, on='make_source', how='left')
    # Merge with standardized model df
    output_df = pd.merge(output_df, df, on=['make_target', 'model_name_source'], how='left')
    output_df = output_df[output_df['mel_id'].notna()]
    # Return
    output_df = output_df.rename(columns={'model_name_target': 'ps_model_name', 'modality_target': 'ps_modality', 'make_target': 'ps_make'})
    return output_df[['company_name','asset_sys_id', 'mel_id', 'make_source', 'ps_make', 'modality_source', 'ps_modality', 'model_name_source', 'ps_model_name', 'model_number']]


if __name__ == "__main__":
    # Get source data
    source_df = get_source_data()
    # Get standardized manufacturer name (removes unmapped manufacturers)
    source_df = standardize_make(source_df)
    # Get target data
    target_df = get_target_data()
    # Loop through mapped manufactuers
    df_list = []
    x = list(source_df['make_target'].unique()) #[0:3]
    for i, make_target in enumerate(x):
        print(f"Processing manufacturer {i} of {len(source_df['make_target'].unique())}")
        this_make_target_df = target_df[target_df['make_target'] == make_target]
        this_make_source_df = source_df[source_df['make_target'] == make_target]
        # Map model
        make_df = map_one_make(this_make_source_df, this_make_target_df)
        if len(make_df) > 0:
            make_df = postprocess_make_df(make_df, this_make_target_df, make_target)
            # Add to list
            make_df.to_csv(f'./batch_files/{str(i)}.csv')        
    # Join batch files
    df = join_batch_files()
    # save to csv
    df.to_csv(MODEL_MODALITY_MAPPING_FILEPATH)
    # Delete batch files
    for file in os.listdir('./batch_files'):
        os.remove(os.path.join('./batch_files', file))
    # Create serialized asset view
    df = create_serialized_asset_view(df)
    # save to csv
    df.to_csv(SERIALIZED_ASSET_VIEW_FILEPATH, index=False)