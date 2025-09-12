import config
import match_functions
import pandas as pd
import os
import re
import llm_model_mapper
from modality_mapping import modality_mapping
from second_stage_modality_mapping import second_stage_modality_mapping
from datetime import datetime


def get_source_data(source_path):
    df = pd.read_csv(source_path, dtype=str, na_filter=False)
    df = df.fillna('').astype(str)
    return df

def get_target_data(filepath_dict):
    # Get MEL data
    df = pd.read_csv(filepath_dict['mel'], dtype=str, na_filter=False)
    df = df.fillna('').astype(str)
    df = df[['New ModelId', 'New Model', 'New Manufacturer', 'New Lvl 2 Category']]
    #df['New ModelId'] = pd.to_numeric(df['New ModelId'], errors='coerce').astype('Int64')
    df = df.rename(columns={'New ModelId': 'mel_id', 'New Manufacturer': 'make_target', 'New Model': 'model_name_target', 'New Lvl 2 Category': 'modality_target'  }) # 'New Lvl 2 Category': 'modality_target' ###
    # Remove empty makes / models
    df = df[(df['make_target'].str.strip() != '') & (df['model_name_target'].str.strip() != '')]
    df = config.remove_duplicates(df, ['make_target', 'model_name_target'], ['mel_id', 'modality_target'])
    return df
    """
    # Get crosswalk data
    if os.path.exists(filepath_dict['model_mapping']):
        df_2 = pd.read_csv(filepath_dict['model_mapping'], dtype=str, na_filter=False)
        df_2 = df_2.fillna('').astype(str)
        #df_2['mel_id'] = pd.to_numeric(df_2['mel_id'], errors='coerce').astype('Int64')
        df_2 = df_2[['mel_id','make_target', 'model_name_source']]
        df_2 = df_2.rename(columns={'model_name_source': 'model_name_target'}) # source name of crosswalk file becomes target name for new data
        df = pd.concat([df, df_2])
    # post process
    df = df[df['make_target'].notna() & df['model_name_target'].notna()]
    df = df[(df['make_target'].str.strip() != '') & (df['model_name_target'].str.strip() != '')]
    # override make (for GEHC -> General Electric and similar cases)
    make_mapping_df = pd.read_csv(filepath_dict['make_mapping'], dtype=str, na_filter=False)
    make_mapping_df = make_mapping_df.fillna('').astype(str)
    df = pd.merge(df, make_mapping_df, on='make_target', how='left').fillna('')
    df = df.drop_duplicates()
    return df """

def get_target_modality_list(mel_path):
    target_df = pd.read_csv(mel_path, dtype=str, na_filter=False)
    target_df = target_df.fillna('').astype(str)
    target_modality_list = list(target_df['New Lvl 2 Category'].unique().astype(str))
    return target_modality_list

def standardize_make(source_df, make_mapping_filepath):
    # Get standardized manufacturer name
    std_df = pd.read_csv(make_mapping_filepath, dtype=str, na_filter=False)
    std_df = std_df.fillna('').astype(str)
    df = pd.merge(source_df, std_df, on='make_source', how='inner').fillna('') # Remove unmapped manufacturers
    df = df.drop(columns=['make_source'])
    return df


def match_model(source_model, target_model_list):
    # Find exact match
    for target_model in target_model_list:
        if source_model.lower() == target_model.lower():
            return target_model, 'exact', 1
    # Find match skipping special characters and spaces
    target_model_dict_list = [{'raw': x, 'skip_special_chars': re.sub(r'[^a-zA-Z0-9]', '', x.lower())} for x in target_model_list]
    source_model = re.sub(r'[^a-zA-Z0-9]', '', source_model.lower())
    for d in target_model_dict_list:
        if source_model == d['skip_special_chars']:
            return d['raw'], 'skip_special_chars', 0.95
    # If no match, return empty string
    return '', 'no_match', 0


def map_one_make(this_make_source_df, this_make_target_df, make_target):
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
    deterministic_unmatched_dl = [x for x in dl if x['model_match_type'] == 'no_match']
    deterministic_matched_dl = [x for x in dl if x['model_match_type'] != 'no_match']
    if len(deterministic_unmatched_dl) > 0 and len(target_model_list) > 0:
        # Use LLM to match unmatched models to target_model_list
        if make_target.strip() != '': # Skip make_target = '' because it's too many models to process and will overload the context iwndow of the LLM
            deterministic_unmatched_dl = llm_model_mapper.main(target_model_list, deterministic_unmatched_dl)
            llm_matched_dl = [x for x in deterministic_unmatched_dl if x['model_name_target'].lower() != 'no_match']
            llm_unmatched_dl = [x for x in deterministic_unmatched_dl if x['model_name_target'].lower() == 'no_match']
            # Post-process unmapped models
            for x in llm_matched_dl: 
                x['model_match_type'] = 'llm'
            for x in llm_unmatched_dl: 
                x['model_match_type'] = 'no_match'
                x['model_name_target'] = ''
                x['model_confidence'] = 0
            deterministic_unmatched_dl = llm_matched_dl + llm_unmatched_dl
        else:
            pass # leave unchanged     
    # Join deterministic and LLM matches and no matches
    dl = deterministic_matched_dl + deterministic_unmatched_dl
    # ADD MODALITY (RAW) by backjoining with source data
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
    this_make_target_df['priority'] = this_make_target_df['mel_id'].apply(lambda x: 0 if x.strip() == '' else 1)
    this_make_target_df = this_make_target_df.sort_values(by='priority', ascending=False)
    this_make_target_df = this_make_target_df.drop_duplicates(subset=['model_name_target'], keep='first')
    # Backfill MEL ID
    make_df = pd.merge(make_df, this_make_target_df, on='model_name_target', how='left').fillna('')
    # Add make 
    make_df['make_target'] = make_target
    # add modality_confidence = 1 if a match is found (because modality comes from MEL; it's the model mapping that is uncertain)
    make_df['modality_confidence'] = make_df['model_match_type'].apply(lambda x: 1 if x != 'no_match' and x.strip() != '' else 0)
    return make_df


def join_batch_files(batch_folder):
    # Get all batch files
    batch_files = [os.path.join(batch_folder, file) for file in os.listdir(batch_folder)]
    # Join all batch files
    df_list = []
    for file in batch_files:
        df = pd.read_csv(file, dtype=str, na_filter=False)
        df = df.fillna('').astype(str)
        df_list.append(df)
    if len(df_list) > 0:
        df = pd.concat(df_list)
    else:
        df = pd.DataFrame()
    return df

def create_serialized_asset_view(df, is_glassbeam, filepath_dict):
    cols=['company_name','make_source', 'model_name_source', 'model_number']
    if not is_glassbeam:
        cols.append('asset_sys_id')
    if is_glassbeam:
        cols.append('product_mgmt_stage')
    # Read source data
    source_df = pd.read_csv(filepath_dict['source'], dtype=str, na_filter=False)
    source_df = source_df.fillna('').astype(str)
    # model number and company name are not required
    if 'model_number' not in source_df.columns:
        source_df['model_number'] = ''
    if 'company_name' not in source_df.columns:
        source_df['company_name'] = ''
    source_df = source_df[cols].fillna('').astype(str)
    #source_df = source_df.drop_duplicates(subset=['asset_sys_id'])
    # Standardize make
    make_mapping_df = pd.read_csv(filepath_dict['make_mapping'], dtype=str, na_filter=False)
    make_mapping_df = make_mapping_df.fillna('').astype(str)
    make_mapping_df = make_mapping_df.rename(columns={'match_type': 'make_match_type', 'confidence': 'make_confidence'})
    output_df = pd.merge(source_df, make_mapping_df, on='make_source', how='left').fillna('')
    # Merge with standardized model df
    output_df = pd.merge(output_df, df, on=['make_target', 'model_name_source'], how='left').fillna('')
    #output_df = output_df[output_df['mel_id'].notna()]
    # Return
    return output_df
    #return output_df[['company_name','asset_sys_id', 'mel_id', 'make_source', 'ps_make', 'modality_source', 'ps_modality', 'model_name_source', 'ps_model_name', 'model_number']]



def add_modality_from_second_stage(row, modality_dl):
    if row['mel_id'] is None or pd.isna(row['mel_id']) or str(row['mel_id']) == 'nan' or str(row['mel_id']).strip() == '':
        if row['modality_target'] is None or pd.isna(row['modality_target']) or str(row['modality_target']) == 'nan' or str(row['modality_target']) == '':
            for d in modality_dl: # d = {'category_raw', 'category_standardized', 'confidence'}
                if d['category_raw'] == row['modality_source']:
                    row['modality_target'] = d['category_standardized']
                    row['modality_confidence'] = d['confidence']
                    row['modality_match_type'] = 'llm_based_on_modality_source'
                    return row
    return row


def override_makes(df, make_override_filepath):
    # Override makes (e.g., Alaris / Carefusion)
    override_df = pd.read_csv(make_override_filepath, dtype=str, na_filter=False)
    df = pd.merge(df, override_df, on='make_target', how='left').fillna('')
    df['make_target'] = df['mel_make_coalesced'].mask(df["mel_make_coalesced"] == "", df["make_target"])
    return df


def remove_preexisting_matches(source_df, model_mapping_filepath):
    if os.path.exists(model_mapping_filepath):
        model_mapping_df = pd.read_csv(model_mapping_filepath, dtype=str, na_filter=False)
        model_mapping_df = model_mapping_df.fillna('').astype(str)
        model_mapping_df = model_mapping_df[['make_target', 'model_name_source', 'mel_id']] #other columns mess up the merging
        if config.OVERRIDE_BLANKS:
            model_mapping_df = model_mapping_df[model_mapping_df['mel_id'].notna()]
        model_mapping_df['existing_record'] = True # all records from model_mapping_df are already evaluated.  the results from previous runs get saved to file
        source_df = pd.merge(source_df, model_mapping_df, on=['make_target', 'model_name_source'], how='left').fillna('')
        source_df = source_df[~(source_df['existing_record'] == True)]
    # Remoe blanks - cannot be matched
    source_df = source_df[source_df['model_name_source'].str.strip() != '']
    return source_df

def print_summary(serialized_df, is_glassbeam):
    source_df = pd.read_csv(filepath_dict['source'], dtype=str, na_filter=False)
    # Source DF
    n_source = len(source_df)
    print(f'N assets (source): {str(n_source)}')
    if is_glassbeam == False:
        n_source_deduped = len(source_df.drop_duplicates(subset=['asset_sys_id']))
    else:
        n_source_deduped = len(source_df.drop_duplicates(subset=['make_source','model_name_source']))
    print(f'N assets (source, deduped): {str(n_source_deduped)}')
    # Serialized asset view - total asset count
    n_serialized = len(serialized_df)
    print(f'N assets (serialized): {str(n_serialized)}')
    if is_glassbeam == False:
        n_serialized_deduped = len(serialized_df.drop_duplicates(subset=['asset_sys_id']))
    else:
        n_serialized_deduped = len(serialized_df.drop_duplicates(subset=['make_source','model_name_source']))
    print(f'N assets (serialized, deduped): {str(n_serialized_deduped)}')
    # Serialized asset view - assets w/ MEL ID
    n_w_mel_id = len(serialized_df[serialized_df['mel_id'].str.strip() != ''])
    print(f'N assets w/ MEL ID (serialized, deduped): {str(n_w_mel_id)}')
    # Serialized asset view - assets w/o MEL ID but with modality
    n_wo_mel_id_with_modality = len(serialized_df[(serialized_df['mel_id'].str.strip() == '') & (serialized_df['modality_target'].str.strip() != '')])
    print(f'N assets w/o MEL ID but with modality (serialized, deduped): {str(n_wo_mel_id_with_modality)}')
    # Serialized asset view - assets w/o MEL ID or modality
    n_with_nothing = len(serialized_df[(serialized_df['mel_id'].str.strip() == '') & (serialized_df['modality_target'].str.strip() == '')])
    print(f'N assets w/o MEL ID or modality (serialized, deduped): {str(n_with_nothing)}')

if __name__ == "__main__":
    source_rump = config.get_source_rump()
    filepath_dict = config.get_filepaths(source_rump)
    # Ask user if it's a glassbeam file
    is_glassbeam_str = input('Is this a glassbeam file? (y/n): ')
    is_glassbeam = True if is_glassbeam_str.lower().strip() == 'y' else False
    # Get source data
    source_df = get_source_data(filepath_dict['source'])
    target_modality_list = get_target_modality_list(filepath_dict['mel'])
    # Get standardized manufacturer name (removes unmapped manufacturers)
    source_df = standardize_make(source_df, filepath_dict['make_mapping'])
    # Get target data
    target_df = get_target_data(filepath_dict)
    # Override makes (e.g., Alaris / Carefusion)
    source_df = override_makes(source_df, filepath_dict['make_override'])
    target_df = override_makes(target_df, filepath_dict['make_override'])
    # Remove preexisting matches - IMPORTANT! This must be done after overriding makes
    source_df = remove_preexisting_matches(source_df, filepath_dict['model_mapping'])
    # Loop through mapped manufactuers
    df_list = []
    x = sorted(list(source_df['make_target'].unique()), reverse=True)
    #Coalesce makes
    for i, make_target in enumerate(x):
        print(f"Processing manufacturer {i} of {len(source_df['make_target'].unique())}: {make_target}")
        this_make_target_df = target_df[target_df['make_target'].str.lower().str.strip() == make_target.lower().strip()]
        this_make_source_df = source_df[source_df['make_target'].str.lower().str.strip() == make_target.lower().strip()]
        # Map model
        make_df = map_one_make(this_make_source_df, this_make_target_df, make_target)
        if len(make_df) > 0:
            make_df = postprocess_make_df(make_df, this_make_target_df, make_target)
            make_df = modality_mapping(make_df, target_modality_list)

            second_stage_modality_mapping_dl = second_stage_modality_mapping(make_df, target_modality_list)
            make_df = make_df.apply(lambda row: add_modality_from_second_stage(row, second_stage_modality_mapping_dl), axis=1)
            make_df = make_df[[x for x in make_df.columns if 'priority' not in x and 'Unnamed:' not in x and x not in ['make_source','match_type']]]

            make_df['added_on'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Add to list
            if not os.path.exists(filepath_dict['batch_folder']):
                os.makedirs(filepath_dict['batch_folder'])
            batch_filepath = filepath_dict['batch_folder'] + f'/{str(i)}.csv'
            make_df.to_csv(batch_filepath)
    if len(x) > 0:
        # Join batch files
        df = join_batch_files(filepath_dict['batch_folder'])
        #Add second stage modality mapping
        df = config.save_new_file(df, filepath_dict['model_mapping'], append_to_old=True, timeout=config.CONCURRENT_WRITE_TIMEOUT_LONG, unique_cols=['make_target', 'model_name_source'], tiebreak_cols=['mel_id','modality_target']) # df contains records from previous runs   
        # Delete batch files
        for file in os.listdir(filepath_dict['batch_folder']):
            os.remove(os.path.join(filepath_dict['batch_folder'], file))
    else:
        df = pd.read_csv(filepath_dict['model_mapping'], dtype=str, na_filter=False)
    # Create output and save to csv (serialized asset view or GB mapping file)
    df = create_serialized_asset_view(df, is_glassbeam, filepath_dict)
    if is_glassbeam:
        unique_cols = ['make_source','model_name_source']
        tiebreak_cols = ['mel_id','modality_target']
    else:
        unique_cols = ['company_name', 'asset_sys_id']
        tiebreak_cols = ['mel_id','modality_target']
    df = df[[x for x in df.columns if 'priority' not in x and 'Unnamed:' not in x]]
    df = config.save_new_file(df, filepath_dict['serialized_asset_view'], append_to_old=False, unique_cols=unique_cols, tiebreak_cols=tiebreak_cols) # don't append. rebuild from scratch based on model_mapping. (model mapping does contain older runs)
    print_summary(df, is_glassbeam)
    
