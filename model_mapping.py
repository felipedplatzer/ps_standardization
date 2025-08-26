import config
import match_functions
import pandas as pd
import os
import re
import llm_model_mapper
from modality_mapping import modality_mapping
from second_stage_modality_mapping import second_stage_modality_mapping
_only_confirmed_matches = True # Set to True to only use matches that are manually confirmed in the model mapping file

def update_mapping_file(df):
    # if mapping file exists, append new matches
    if os.path.exists(config.MAKE_MAPPING_FILEPATH):
        df_old = pd.read_csv(config.MAKE_MAPPING_FILEPATH)
        df = pd.concat([df_old, df])
    # save to mapping file
    df.to_csv(config.MAKE_MAPPING_FILEPATH, index=False)


def get_source_data(source_path):
    df = pd.read_csv(source_path)
    df = df.fillna('').astype(str)
    return df

def get_target_data():
    # Get MEL data
    df = pd.read_csv(config.MEL_PATH)
    df = df[['New ModelId', 'New Model', 'New Manufacturer', 'New Lvl 2 Category']]
    df['New ModelId'] = pd.to_numeric(df['New ModelId'], errors='coerce').astype('Int64')
    df = df.rename(columns={'New ModelId': 'mel_id', 'New Manufacturer': 'make_target', 'New Model': 'model_name_target', 'New Lvl 2 Category': 'modality_target'  }) # 'New Lvl 2 Category': 'modality_target' ###
    # Get crosswalk data
    if os.path.exists(config.MODEL_MAPPING_FILEPATH):
        df_2 = pd.read_csv(config.MODEL_MAPPING_FILEPATH)
        df_2['mel_id'] = pd.to_numeric(df_2['mel_id'], errors='coerce').astype('Int64')
        df_2 = df_2[['mel_id','make_target', 'model_name_source', 'confirmed']]
        df_2 = df_2.rename(columns={'model_name_source': 'model_name_target'}) # source name of crosswalk file becomes target name for new data
        # Get only confirmed records
        if _only_confirmed_matches: 
            df_2 = df_2[df_2['confirmed'] == True]
        df = pd.concat([df, df_2])
    # post process
    df = df[df['make_target'].notna() & df['model_name_target'].notna()]
    # override make (for GEHC -> General Electric and similar cases)
    df['make_target'] = df['make_target'].apply(lambda x: config.make_override(x))
    df = df.drop_duplicates()
    return df

def get_target_modality_list():
    target_df = pd.read_csv(config.MEL_PATH)
    target_modality_list = list(target_df['New Lvl 2 Category'].unique().astype(str))
    return target_modality_list

def standardize_make(source_df):
    # Get standardized manufacturer name
    std_df = pd.read_csv(config.MAKE_MAPPING_FILEPATH)
    df = pd.merge(source_df, std_df, on='make_source', how='inner') # Remove unmapped manufacturers
    df = df.drop(columns=['make_source'])
    return df


def match_model(source_model, target_model_list):
    # Find exact match
    for target_model in target_model_list:
        if source_model.lower() == target_model.lower():
            return target_model, 'exact', 1
    # Find match skipping special characters and spaces
    target_model_list = [re.sub(r'[^a-zA-Z0-9]', '', x.lower()) for x in target_model_list]
    source_model = re.sub(r'[^a-zA-Z0-9]', '', source_model.lower())
    for target_model in target_model_list:
        if source_model == target_model:
            return target_model, 'skip_special_chars', 0.95
    # If no match, return empty string
    return '', 'no_match', 0


def map_one_make(this_make_source_df, this_make_target_df):
    this_make_original_dl = this_make_source_df.to_dict(orient='records') #needed to add raw modality
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
        target_model, match_type, match_score = match_model(source_model, target_model_list)
        match_dict = {'model_name_source': source_model, 'model_name_target': target_model, 'model_match_type': match_type, 'model_confidence': match_score}
        dl.append(match_dict)
    # LLM mapping
    # Batch all remaining, unmatched models
    unmatched_dl = [x for x in dl if x['model_match_type'] == 'no_match']
    deterministic_matched_dl = [x for x in dl if x['model_match_type'] != 'no_match']
    if len(unmatched_dl) > 0 and len(target_model_list) > 0:
        # Use LLM to match unmatched models to target_model_list
        llm_dl = llm_model_mapper.main(target_model_list, unmatched_dl)
        # Post-process unmapped models
        llm_matched_dl = [x for x in llm_dl if x['model_name_target'].lower() != 'no_match']
        for x in llm_matched_dl: 
            x['model_match_type'] = 'llm'
        unmatched_dl = [x for x in llm_dl if x['model_name_target'].lower() == 'no_match']
        for x in unmatched_dl: 
            x['model_match_type'] = 'no_match'
            x['model_name_target'] = ''
            x['model_confidence'] = 0
    else:
        llm_matched_dl = []
    # Join deterministic and LLM matches and no matches
    dl = deterministic_matched_dl + llm_matched_dl + unmatched_dl
    # ADD MODALITY (RAW)
    for d in dl:
        for x in this_make_original_dl:
            if x['model_name_source'] == d['model_name_source']:
                d['modality_source'] = x['modality_source']
                break
    # convert to dataframe
    df = pd.DataFrame(dl)
    if 'modality_source' not in df.columns:
        df['modality_source'] = ''
    else:
        df['modality_source'] = df['modality_source'].fillna('')
    return df

def postprocess_make_df(make_df, this_make_target_df, make_target):
    # Remove duplicates
    this_make_target_df = this_make_target_df.drop_duplicates(subset=['model_name_target'])
    # Backfill MEL ID
    make_df = pd.merge(make_df, this_make_target_df, on='model_name_target', how='left')
    # Add make 
    make_df['make_target'] = make_target
    # add match type
    make_df['confirmed'] = make_df['model_match_type'].apply(lambda x: True if x == 'exact' else False)
    # add modality_confidence = 1 if a match is found (because modality comes from MEL; it's the model mapping that is uncertain)
    make_df['modality_confidence'] = make_df['model_match_type'].apply(lambda x: 1 if x != 'no_match' and x.strip() != '' else 0)
    return make_df


def join_batch_files():
    # Get all batch files
    batch_files = [os.path.join(config.BATCH_FILES_PATH, file) for file in os.listdir(config.BATCH_FILES_PATH)]
    # Join all batch files
    df = pd.concat([pd.read_csv(file) for file in batch_files])
    return df

def create_serialized_asset_view(df, is_glassbeam):
    cols=['company_name','make_source', 'model_name_source', 'model_number']
    if not is_glassbeam:
        cols.append('asset_sys_id')
    # Read source data
    source_df = pd.read_csv(config.SOURCE_PATH)
    # model number and company name are not required
    if 'model_number' not in source_df.columns:
        source_df['model_number'] = ''
    if 'company_name' not in source_df.columns:
        source_df['company_name'] = ''
    source_df = source_df[cols].fillna('').astype(str)
    #source_df = source_df.drop_duplicates(subset=['asset_sys_id'])
    # Standardize make
    make_mapping_df = pd.read_csv(config.MAKE_MAPPING_FILEPATH)
    make_mapping_df = make_mapping_df.rename(columns={'confirmed': 'make_confirmed', 'match_type': 'make_match_type', 'confidence': 'make_confidence'})
    output_df = pd.merge(source_df, make_mapping_df, on='make_source', how='left')
    # Merge with standardized model df
    df = df.rename(columns={'confirmed': 'model_confirmed'})
    output_df = pd.merge(output_df, df, on=['make_target', 'model_name_source'], how='left')
    #output_df = output_df[output_df['mel_id'].notna()]
    # Return
    output_df = output_df.rename(columns={'model_name_target': 'ps_model_name', 'modality_target': 'ps_modality', 'make_target': 'ps_make'})
    return output_df
    #return output_df[['company_name','asset_sys_id', 'mel_id', 'make_source', 'ps_make', 'modality_source', 'ps_modality', 'model_name_source', 'ps_model_name', 'model_number']]

def add_glassbeam_columns(df):
    #Merge with make mapping df to add make_target
    source_df = pd.read_csv(config.SOURCE_PATH)
    make_mapping_df = pd.read_csv(config.MAKE_MAPPING_FILEPATH)
    output_df = pd.merge(source_df, make_mapping_df, on='make_source', how='left')
    output_df = output_df.drop(columns=['full', 'partial', 'codev'])
    # Add columns full, partial, codev from source date
    output_df = pd.merge(df, output_df, on=['make_target', 'model_name_source'], how='left')
    return output_df

def add_modality_from_second_stage(row, modality_dl):
    if row['mel_id'] is None or pd.isna(row['mel_id']) or str(row['mel_id']) == 'nan' or str(row['mel_id']) == '':
        if row['modality_target'] is None or pd.isna(row['modality_target']) or str(row['modality_target']) == 'nan' or str(row['modality_target']) == '':
            for d in modality_dl: # d = {'category_raw', 'category_standardized', 'confidence'}
                if d['category_raw'] == row['modality_source']:
                    row['modality_target'] = d['category_standardized']
                    row['modality_confidence'] = d['confidence']
                    row['modality_match_type'] = 'llm_based_on_modality_source'
                    return row
    return row

if __name__ == "__main__":
    # Ask user if it's a glassbeam file
    is_glassbeam_str = input('Is this a glassbeam file? (y/n): ')
    is_glassbeam = True if is_glassbeam_str.lower().strip() == 'y' else False
    # Get source data
    source_df = get_source_data(config.SOURCE_PATH)
    target_modality_list = get_target_modality_list()
    # Get standardized manufacturer name (removes unmapped manufacturers)
    source_df = standardize_make(source_df)
    # Get target data
    target_df = get_target_data()
    # Loop through mapped manufactuers
    df_list = []
    source_df['make_target'] = source_df['make_target'].fillna('')
    x = sorted(list(source_df['make_target'].unique()), reverse=True) 
    
    for i, make_target in enumerate(x):
        print(f"Processing manufacturer {i} of {len(source_df['make_target'].unique())}: {make_target}")
        this_make_target_df = target_df[target_df['make_target'].str.lower().str.strip() == make_target.lower().strip()]
        this_make_source_df = source_df[source_df['make_target'].str.lower().str.strip() == make_target.lower().strip()]
        # Map model
        make_df = map_one_make(this_make_source_df, this_make_target_df)
        if len(make_df) > 0:
            batch_filepath = config.BATCH_FILES_PATH + f'/{str(i)}.csv'
            make_df = postprocess_make_df(make_df, this_make_target_df, make_target)
            make_df = modality_mapping(make_df, target_modality_list)
            # Add to list
            make_df.to_csv(batch_filepath)        
    
    # Join batch files
    df = join_batch_files()
    # save to csv
    config.save_new_file(df, config.MODEL_MAPPING_FILEPATH)
    # Delete batch files
    for file in os.listdir(config.BATCH_FILES_PATH):
        os.remove(os.path.join(config.BATCH_FILES_PATH, file))
    #Add second stage modality mapping
    second_stage_modality_mapping_dl = second_stage_modality_mapping(df, target_modality_list)
    df = df.apply(lambda row: add_modality_from_second_stage(row, second_stage_modality_mapping_dl), axis=1)
    config.save_new_file(df, config.MODEL_MAPPING_FILEPATH)
    # Create output and save to csv (serialized asset view or GB mapping file)
    df = create_serialized_asset_view(df, is_glassbeam)
    config.save_new_file(df, config.SERIALIZED_ASSET_VIEW_FILEPATH)
    