import config
import match_functions
import pandas as pd
import os
import re
import atexit
import signal
import llm_model_mapper
from hierarchical_modality_mapper import build_taxonomy_lookup, map_unmatched_devices
from validation_tests import run_validation_tests
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Global variable to track current batch folder for cleanup
_current_batch_info = {
    'batch_folder': None,
    'model_mapping_path': None,
    'cleanup_done': False
}


def _cleanup_batch_files():
    """
    Cleanup function to save batch files to model_mapping.csv on exit/crash.
    Called automatically via atexit or signal handlers.
    """
    global _current_batch_info
    
    if _current_batch_info['cleanup_done']:
        return
    
    batch_folder = _current_batch_info.get('batch_folder')
    model_mapping_path = _current_batch_info.get('model_mapping_path')
    
    if batch_folder and os.path.exists(batch_folder):
        batch_files = [os.path.join(batch_folder, f) for f in os.listdir(batch_folder) if f.endswith('.xlsx')]
        
        if batch_files:
            print(f'\n{"="*60}')
            print('CLEANUP: Saving {0} batch files to model_mapping.csv before exit...'.format(len(batch_files)))
            print(f'{"="*60}')
            
            try:
                # Join batch files
                df_list = []
                for file in batch_files:
                    try:
                        df = pd.read_excel(file, dtype=str, na_filter=False)
                        df = df.fillna('').astype(str)
                        df = df.apply(lambda x: x.str.strip())
                        df_list.append(df)
                    except Exception as e:
                        print(f'Warning: Could not read batch file {file}: {e}')
                
                if df_list:
                    combined_df = pd.concat(df_list, ignore_index=True)
                    
                    # Save to model_mapping.csv
                    if model_mapping_path:
                        combined_df = config.save_new_file(
                            combined_df,
                            model_mapping_path,
                            append_to_old=True,
                            timeout=config.CONCURRENT_WRITE_TIMEOUT_LONG,
                            unique_cols=['make_target_normalized', 'model_name_source_normalized'],
                            tiebreak_cols=['mel_id', 'l3_modality_target'],
                            print_stats=True
                        )
                        print(f'Successfully saved {len(combined_df)} records to {model_mapping_path}')
                        
                        # Remove batch files after successful save
                        for file in batch_files:
                            try:
                                os.remove(file)
                            except Exception as e:
                                print(f'Warning: Could not remove batch file {file}: {e}')
                        print(f'Removed {len(batch_files)} batch files')
                    
            except Exception as e:
                print(f'Error during cleanup: {e}')
                print('Batch files have been preserved in:', batch_folder)
    
    _current_batch_info['cleanup_done'] = True


def _signal_handler(signum, frame):
    """Handle interrupt signals by running cleanup."""
    print(f'\nReceived signal {signum}, running cleanup...')
    _cleanup_batch_files()
    exit(1)


# Register cleanup handlers
atexit.register(_cleanup_batch_files)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def add_normalized_columns_source(df):
    """
    Add normalized columns to source dataframe.
    Normalizes: model_name_source, model_number, l1/l2/l3_modality_source
    """
    if 'model_name_source' in df.columns:
        df['model_name_source_normalized'] = df['model_name_source'].apply(config.normalize_name)
    if 'model_number' in df.columns:
        df['model_number_normalized'] = df['model_number'].apply(config.normalize_name)
    
    # Normalize L1, L2, L3 modality source fields
    for level in ['l1', 'l2', 'l3']:
        col = f'{level}_modality_source'
        if col in df.columns:
            df[f'{col}_normalized'] = df[col].apply(config.normalize_name)
    
    return df


def add_normalized_columns_target(df):
    """
    Add normalized columns to target (MEL) dataframe.
    Normalizes: make_target, model_name_target, l1/l2/l3_modality_target
    """
    if 'make_target' in df.columns:
        df['make_target_normalized'] = df['make_target'].apply(config.normalize_name)
    if 'model_name_target' in df.columns:
        df['model_name_target_normalized'] = df['model_name_target'].apply(config.normalize_name)
    
    # Normalize L1, L2, L3 modality target fields
    for level in ['l1', 'l2', 'l3']:
        col = f'{level}_modality_target'
        if col in df.columns:
            df[f'{col}_normalized'] = df[col].apply(config.normalize_name)
    
    return df


def get_source_data(source_path):
    """Load and normalize source data with L1/L2/L3 modality fields."""
    df = pd.read_excel(source_path, dtype=str, na_filter=False)
    df = df.fillna('').astype(str)
    df = df.apply(lambda x: x.str.strip())
    
    # Map source modality columns using config
    source_cols = config.SOURCE_COLUMNS
    for level in ['l1', 'l2', 'l3']:
        config_key = f'{level}_modality_source'
        if config_key in source_cols:
            source_col_name = source_cols[config_key]
            if source_col_name in df.columns:
                df[f'{level}_modality_source'] = df[source_col_name]
            else:
                df[f'{level}_modality_source'] = ''
        else:
            df[f'{level}_modality_source'] = ''
    
    # Add normalized columns for source fields
    df = add_normalized_columns_source(df)
    return df


def get_target_data(filepath_dict):
    """Load and normalize MEL data with L1/L2/L3 modality fields."""
    df = pd.read_excel(filepath_dict['mel'], dtype=str, na_filter=False)
    df = df.fillna('').astype(str)
    df = df.apply(lambda x: x.str.strip())
    
    # Use MEL_COLUMNS config for column name mapping
    mel_cols = config.MEL_COLUMNS
    
    # Build required columns list
    required_cols = [mel_cols['mel_id'], mel_cols['model_name_target'], mel_cols['make_target']]
    for level in ['l1', 'l2', 'l3']:
        key = f'{level}_modality_target'
        if key in mel_cols:
            required_cols.append(mel_cols[key])
    
    df = df[required_cols]
    
    # Rename columns
    rename_map = {
        mel_cols['mel_id']: 'mel_id', 
        mel_cols['make_target']: 'make_target', 
        mel_cols['model_name_target']: 'model_name_target',
    }
    for level in ['l1', 'l2', 'l3']:
        key = f'{level}_modality_target'
        if key in mel_cols:
            rename_map[mel_cols[key]] = f'{level}_modality_target'
    
    df = df.rename(columns=rename_map)
    
    # Add normalized columns for target fields
    df = add_normalized_columns_target(df)
    
    # Remove empty makes / models (check normalized versions)
    df = df[(df['make_target_normalized'].str.strip() != '') & (df['model_name_target_normalized'].str.strip() != '')]
    df = config.remove_duplicates(df, ['make_target_normalized', 'model_name_target_normalized'], ['mel_id', 'l3_modality_target'])
    
    return df


def get_taxonomy(filepath_dict):
    """Build taxonomy lookup from MEL data."""
    target_df = get_target_data(filepath_dict)
    taxonomy = build_taxonomy_lookup(target_df)
    return taxonomy


def standardize_make(source_df, make_mapping_filepath):
    """
    Get standardized manufacturer name using normalized make_source.
    Maps make_source_normalized -> make_source_normalized (from make_mapping) -> get make_target and make_target_normalized
    """
    std_df = pd.read_excel(make_mapping_filepath, dtype=str, na_filter=False)
    std_df = std_df.fillna('').astype(str)
    std_df = std_df.apply(lambda x: x.str.strip())
    
    # Add make_source_normalized to source_df if not present
    if 'make_source_normalized' not in source_df.columns:
        source_df['make_source_normalized'] = source_df['make_source'].apply(config.normalize_name)
    
    # Check if mapping file has normalized columns (new format)
    if 'make_source_normalized' in std_df.columns and 'make_target_normalized' in std_df.columns:
        cols_to_keep = ['make_source', 'make_source_normalized', 'make_target', 'make_target_normalized']
        # Include manufacturer_aliases if present
        if 'manufacturer_aliases' in std_df.columns:
            cols_to_keep.append('manufacturer_aliases')
        std_df = std_df[cols_to_keep]
        std_df = std_df.drop_duplicates(subset=['make_source_normalized'])
        df = pd.merge(source_df, std_df, on='make_source_normalized', how='inner', suffixes=('', '_mapping')).fillna('')
        if 'make_source_mapping' in df.columns:
            df = df.drop(columns=['make_source_mapping'])
    else:
        cols_to_keep = ['make_source', 'make_target']
        if 'manufacturer_aliases' in std_df.columns:
            cols_to_keep.append('manufacturer_aliases')
        std_df = std_df[cols_to_keep]
        std_df = std_df.drop_duplicates(subset=['make_source'])
        df = pd.merge(source_df, std_df, on='make_source', how='inner').fillna('')
        df['make_target_normalized'] = df['make_target'].apply(config.normalize_name)
    
    df = df.drop(columns=['make_source'], errors='ignore')
    return df


def match_model_normalized(source_model_normalized, target_model_normalized_list):
    """
    Match source model (normalized) to target model list (normalized).
    Returns: (matched_target_normalized, match_type, confidence)
    """
    for target_normalized in target_model_normalized_list:
        if source_model_normalized == target_normalized:
            return target_normalized, 'exact', 1
    
    source_clean = re.sub(r'[^A-Z0-9]', '', source_model_normalized)
    for target_normalized in target_model_normalized_list:
        target_clean = re.sub(r'[^A-Z0-9]', '', target_normalized)
        if source_clean == target_clean:
            return target_normalized, 'skip_special_chars', 0.95
    
    return '', 'no_match', 0


def map_one_make(this_make_source_df, this_make_target_df, make_target_normalized, make_target):
    """
    Map models for one manufacturer using normalized fields.
    """
    # Collect source modality columns
    source_modality_cols = []
    for level in ['l1', 'l2', 'l3']:
        col = f'{level}_modality_source'
        norm_col = f'{col}_normalized'
        if col in this_make_source_df.columns:
            source_modality_cols.append(col)
        if norm_col in this_make_source_df.columns:
            source_modality_cols.append(norm_col)
    
    base_cols = ['model_name_source', 'model_name_source_normalized'] + source_modality_cols
    base_cols = [c for c in base_cols if c in this_make_source_df.columns]
    
    if make_target_normalized.strip() == '':
        output_df = this_make_source_df[base_cols].copy()
        output_df['model_name_target'] = ''
        output_df['model_name_target_normalized'] = ''
        output_df['model_match_type'] = 'skipped_blank_make'
        output_df['model_confidence'] = 0
        return output_df
    else:
        this_make_original_dl = this_make_source_df.to_dict(orient='records')
        dl = []
        
        # Get source and target model lists
        source_model_pairs = this_make_source_df[['model_name_source', 'model_name_source_normalized']].drop_duplicates()
        source_model_pairs = source_model_pairs[source_model_pairs['model_name_source_normalized'].str.strip() != '']
        
        target_model_pairs = this_make_target_df[['model_name_target', 'model_name_target_normalized']].drop_duplicates()
        target_model_pairs = target_model_pairs[target_model_pairs['model_name_target_normalized'].str.strip() != '']
        
        target_normalized_to_original = dict(zip(target_model_pairs['model_name_target_normalized'], target_model_pairs['model_name_target']))
        target_model_normalized_list = list(target_model_pairs['model_name_target_normalized'].unique())
        
        # Deterministic mapping
        for _, row in source_model_pairs.iterrows():
            source_model = row['model_name_source']
            source_model_normalized = row['model_name_source_normalized']
            
            target_normalized, match_type, match_score = match_model_normalized(source_model_normalized, target_model_normalized_list)
            target_model = target_normalized_to_original.get(target_normalized, '')
            
            match_dict = {
                'model_name_source': source_model,
                'model_name_source_normalized': source_model_normalized,
                'model_name_target': target_model,
                'model_name_target_normalized': target_normalized,
                'model_match_type': match_type,
                'model_confidence': match_score
            }
            dl.append(match_dict)
        
        # LLM mapping for unmatched models
        deterministic_unmatched_dl = [x for x in dl if x['model_match_type'] == 'no_match']
        deterministic_matched_dl = [x for x in dl if x['model_match_type'] != 'no_match']
        
        if len(deterministic_unmatched_dl) > 0 and len(target_model_normalized_list) > 0:
            if make_target_normalized.strip() != '':
                target_model_list = list(target_model_pairs['model_name_target'].unique())
                
                # Create a lookup to preserve original data (including model_name_source_normalized)
                original_data_lookup = {x['model_name_source']: x.copy() for x in deterministic_unmatched_dl}
                
                # Call LLM mapper
                llm_results = llm_model_mapper.main(target_model_list, deterministic_unmatched_dl)
                
                # Merge LLM results back with original data to preserve model_name_source_normalized
                merged_results = []
                for llm_result in llm_results:
                    model_name_source = llm_result.get('model_name_source', '')
                    original_data = original_data_lookup.get(model_name_source, {})
                    
                    # Start with original data (preserves model_name_source_normalized)
                    merged_record = original_data.copy()
                    
                    # Update with LLM results
                    merged_record['model_name_target'] = llm_result.get('model_name_target', '')
                    merged_record['model_confidence'] = llm_result.get('model_confidence', 0)
                    
                    merged_results.append(merged_record)
                
                deterministic_unmatched_dl = merged_results
                
                llm_matched_dl = [x for x in deterministic_unmatched_dl if str(x.get('model_name_target', '')).lower() != 'no_match' and str(x.get('model_name_target', '')).strip() != '']
                llm_unmatched_dl = [x for x in deterministic_unmatched_dl if str(x.get('model_name_target', '')).lower() == 'no_match' or str(x.get('model_name_target', '')).strip() == '']
                
                for x in llm_matched_dl:
                    x['model_match_type'] = 'llm'
                    x['model_name_target_normalized'] = config.normalize_name(x.get('model_name_target', ''))
                
                for x in llm_unmatched_dl:
                    x['model_match_type'] = 'no_match'
                    x['model_name_target'] = ''
                    x['model_name_target_normalized'] = ''
                    x['model_confidence'] = 0
                
                deterministic_unmatched_dl = llm_matched_dl + llm_unmatched_dl
        
        dl = deterministic_matched_dl + deterministic_unmatched_dl
        
        # Add source modalities by backjoining with source data
        for d in dl:
            for x in this_make_original_dl:
                if x['model_name_source'] == d['model_name_source']:
                    for level in ['l1', 'l2', 'l3']:
                        col = f'{level}_modality_source'
                        norm_col = f'{col}_normalized'
                        d[col] = x.get(col, '')
                        d[norm_col] = x.get(norm_col, '')
                    break
        
        df = pd.DataFrame(dl)
        
        # Ensure all modality columns exist
        for level in ['l1', 'l2', 'l3']:
            for suffix in ['_modality_source', '_modality_source_normalized']:
                col = f'{level}{suffix}'
                if col not in df.columns:
                    df[col] = ''
                else:
                    df[col] = df[col].fillna('')
        
        if 'model_name_target_normalized' not in df.columns:
            df['model_name_target_normalized'] = ''
        
        return df


def postprocess_make_df(make_df, this_make_target_df, make_target, make_target_normalized):
    """
    Postprocess to add mel_id and L1/L2/L3 modalities from MEL.
    Uses normalized model_name_target for joining.
    For unmatched models, fills model_name_target with model_name_source and l3_modality_target with l3_modality_source.
    """
    target_for_join = this_make_target_df.copy()
    target_for_join['priority'] = target_for_join['mel_id'].apply(lambda x: 0 if str(x).strip() == '' else 1)
    target_for_join = target_for_join.sort_values(by='priority', ascending=False)
    target_for_join = target_for_join.drop_duplicates(subset=['model_name_target_normalized'], keep='first')
    
    # Select columns for joining - include L1, L2, L3 modalities
    join_cols = ['model_name_target_normalized', 'mel_id']
    for level in ['l1', 'l2', 'l3']:
        col = f'{level}_modality_target'
        norm_col = f'{col}_normalized'
        if col in target_for_join.columns:
            join_cols.append(col)
        if norm_col in target_for_join.columns:
            join_cols.append(norm_col)
    
    target_for_join = target_for_join[join_cols]
    
    # Backfill MEL ID and modalities
    make_df = pd.merge(make_df, target_for_join, on='model_name_target_normalized', how='left').fillna('')
    
    # Add make
    make_df['make_target'] = make_target
    make_df['make_target_normalized'] = make_target_normalized
    
    # For unmatched models: fill target fields with source values
    # Fill model_name_target with model_name_source if no match found
    unmatched_mask = make_df['model_match_type'].str.lower() == 'no_match'
    
    if 'model_name_target' in make_df.columns:
        make_df.loc[unmatched_mask & (make_df['model_name_target'].str.strip() == ''), 'model_name_target'] = \
            make_df.loc[unmatched_mask & (make_df['model_name_target'].str.strip() == ''), 'model_name_source']
    
    if 'model_name_target_normalized' in make_df.columns:
        make_df.loc[unmatched_mask & (make_df['model_name_target_normalized'].str.strip() == ''), 'model_name_target_normalized'] = \
            make_df.loc[unmatched_mask & (make_df['model_name_target_normalized'].str.strip() == ''), 'model_name_source_normalized']
    
    # Fill l3_modality_target with l3_modality_source if no match found
    if 'l3_modality_target' in make_df.columns and 'l3_modality_source' in make_df.columns:
        make_df.loc[unmatched_mask & (make_df['l3_modality_target'].str.strip() == ''), 'l3_modality_target'] = \
            make_df.loc[unmatched_mask & (make_df['l3_modality_target'].str.strip() == ''), 'l3_modality_source']
    
    if 'l3_modality_target_normalized' in make_df.columns and 'l3_modality_source_normalized' in make_df.columns:
        make_df.loc[unmatched_mask & (make_df['l3_modality_target_normalized'].str.strip() == ''), 'l3_modality_target_normalized'] = \
            make_df.loc[unmatched_mask & (make_df['l3_modality_target_normalized'].str.strip() == ''), 'l3_modality_source_normalized']
    
    # Add modality confidence - 1 if a match is found (modality comes from MEL)
    make_df['modality_confidence'] = make_df['model_match_type'].apply(lambda x: 1 if x != 'no_match' and str(x).strip() != '' else 0)
    
    # Add verified_model_name - 1 if a model match was found, 0 if not
    make_df['verified_model_name'] = make_df['model_match_type'].apply(lambda x: 1 if str(x).strip().lower() != 'no_match' and str(x).strip() != '' else 0)
    
    return make_df


def join_batch_files(batch_files):
    """Join all batch files into a single dataframe."""
    df_list = []
    for file in batch_files:
        df = pd.read_excel(file, dtype=str, na_filter=False)
        df = df.fillna('').astype(str)
        df = df.apply(lambda x: x.str.strip())
        df_list.append(df)
    df = pd.concat(df_list)
    return df


def get_all_customers_filepath():
    """Get the filepath for the consolidated all-customers serialized asset view."""
    return './files/serialized_asset_views/serialized_asset_view_all_customers.xlsx'


def clean_serialized_df(df, is_glassbeam):
    """
    Clean the serialized asset view dataframe:
    1. Remove unwanted columns
    2. Uppercase and trim company_name and asset_sys_id
    """
    # Columns to remove from serialized files
    cols_to_remove = [
        'model_confidence',
        'modality_confidence',
        'l1_modality_confidence',
        'l2_modality_confidence',
        'l3_modality_confidence',
        'make_source_mapping',
        'matched_via_alias',
        'model_name_source_model',
        'l1_modality_source_model',
        'l2_modality_source_model',
        'l3_modality_source_model',
        'l1_modality_source_normalized_model',
        'l2_modality_source_normalized_model',
        'l3_modality_source_normalized_model',
        'make_target_model',
    ]
    
    # Remove unwanted columns
    df = df.drop(columns=[col for col in cols_to_remove if col in df.columns], errors='ignore')
    
    # Uppercase and trim company_name and asset_sys_id (for deduplication and storage)
    if not is_glassbeam:
        if 'company_name' in df.columns:
            df['company_name'] = df['company_name'].apply(lambda x: str(x).strip().upper() if pd.notna(x) else '')
        if 'asset_sys_id' in df.columns:
            df['asset_sys_id'] = df['asset_sys_id'].apply(lambda x: str(x).strip().upper() if pd.notna(x) else '')
    
    return df


def append_to_all_customers_file(df, is_glassbeam):
    """
    Append records to the consolidated serialized_asset_view_all_customers file.
    If a record with the same company_name and asset_sys_id already exists, 
    it will be replaced with the new record.
    
    Args:
        df: DataFrame with new records to append
        is_glassbeam: Boolean indicating if this is a glassbeam file
    """
    all_customers_filepath = get_all_customers_filepath()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(all_customers_filepath), exist_ok=True)
    
    # Define unique columns based on file type
    if is_glassbeam:
        unique_cols = ['make_source', 'model_name_source']
    else:
        unique_cols = ['company_name', 'asset_sys_id']
    
    # Check if all unique columns exist in df
    missing_cols = [col for col in unique_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Cannot append to all_customers file - missing columns: {missing_cols}")
        return
    
    if os.path.exists(all_customers_filepath):
        # Read existing file
        existing_df = pd.read_excel(all_customers_filepath, dtype=str, na_filter=False)
        existing_df = existing_df.fillna('').astype(str)
        existing_df = existing_df.apply(lambda x: x.str.strip())
        
        # Uppercase and trim unique columns in existing file for consistent comparison
        if not is_glassbeam:
            if 'company_name' in existing_df.columns:
                existing_df['company_name'] = existing_df['company_name'].apply(lambda x: str(x).strip().upper() if pd.notna(x) else '')
            if 'asset_sys_id' in existing_df.columns:
                existing_df['asset_sys_id'] = existing_df['asset_sys_id'].apply(lambda x: str(x).strip().upper() if pd.notna(x) else '')
        
        print(f'\nAppending to all_customers file...')
        print(f'  Existing records: {len(existing_df)}')
        print(f'  New records to add: {len(df)}')
        
        # Create a composite key for comparison (using already uppercased/trimmed values)
        def create_key(row):
            return '|'.join([str(row.get(col, '')).strip().upper() for col in unique_cols])
        
        existing_df['_key'] = existing_df.apply(create_key, axis=1)
        df['_key'] = df.apply(create_key, axis=1)
        
        # Remove existing records that will be replaced by new records
        new_keys = set(df['_key'].unique())
        existing_df = existing_df[~existing_df['_key'].isin(new_keys)]
        
        # Drop the key column
        existing_df = existing_df.drop(columns=['_key'])
        df = df.drop(columns=['_key'])
        
        # Combine existing and new
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        
        # Remove any remaining duplicates (keep last occurrence = new record)
        combined_df = combined_df.drop_duplicates(subset=unique_cols, keep='last')
        
        print(f'  Records after deduplication: {len(combined_df)}')
    else:
        combined_df = df.copy()
        combined_df = combined_df.drop_duplicates(subset=unique_cols, keep='last')
        print(f'\nCreating new all_customers file with {len(combined_df)} records')
    
    # Remove unwanted columns
    cols_to_drop = [col for col in combined_df.columns if 'Unnamed:' in col or 'priority' in col]
    combined_df = combined_df.drop(columns=cols_to_drop, errors='ignore')
    
    # Clean serialized dataframe (remove unwanted cols, ensure identifiers are uppercased/trimmed)
    combined_df = clean_serialized_df(combined_df, is_glassbeam)
    
    # Reorder columns for consistency
    combined_df = reorder_output_columns(combined_df, is_glassbeam)
    
    # Save the consolidated file
    combined_df.to_excel(all_customers_filepath, index=False, na_rep='')
    print(f'  Saved to: {all_customers_filepath}')


def reorder_output_columns(df, is_glassbeam):
    """
    Reorder columns in the output DataFrame.
    Order: 
    1. Identifiers (company_name, asset_sys_id)
    2. Source fields: make, model, modality_l1, l2, l3
    3. Normalized source fields: make, model, modality_l1, l2, l3 (only if they exist)
    4. Target fields: mel_id, make, model, modality_l1, l2, l3
    5. Normalized target fields: make, model, modality_l1, l2, l3 (only if they exist)
    6. Match type and confidence columns
    7. Other columns
    8. added_on at the end
    """
    # Define column order groups
    id_cols = ['company_name']
    if not is_glassbeam:
        id_cols.append('asset_sys_id')
    # product_mgmt_stage moved to end_cols
    
    source_cols = [
        'make_source',
        'model_name_source',
        'model_number',
        'l1_modality_source',
        'l2_modality_source',
        'l3_modality_source',
    ]
    
    # Normalized source columns (between source and target)
    normalized_source_cols = [
        'make_source_normalized',
        'model_name_source_normalized',
        'model_number_normalized',
        'l1_modality_source_normalized',
        'l2_modality_source_normalized',
        'l3_modality_source_normalized',
    ]
    
    target_cols = [
        'mdm_model_id',
        'verified_model_name',
        'make_target',
        'manufacturer_aliases',
        'model_name_target',
        'l1_modality_target',
        'l2_modality_target',
        'l3_modality_target',
    ]
    
    # Normalized target columns (after target)
    normalized_target_cols = [
        'make_target_normalized',
        'model_name_target_normalized',
        'l1_modality_target_normalized',
        'l2_modality_target_normalized',
        'l3_modality_target_normalized',
    ]
    
    match_cols = [
        'make_match_type',
        'make_confidence',
        'model_match_type',
        'modality_match_type',
    ]
    
    # Columns that should be at the end
    # Columns at the end: product_mgmt_stage before added_on
    end_cols = ['product_mgmt_stage', 'added_on']
    
    # Build ordered column list
    ordered_cols = []
    
    # Add columns in order, only if they exist in the dataframe
    for col_group in [id_cols, source_cols, normalized_source_cols, target_cols, normalized_target_cols, match_cols]:
        for col in col_group:
            if col in df.columns and col not in ordered_cols:
                ordered_cols.append(col)
    
    # Add any remaining columns (except end_cols)
    for col in df.columns:
        if col not in ordered_cols and col not in end_cols:
            ordered_cols.append(col)
    
    # Add end columns
    for col in end_cols:
        if col in df.columns:
            ordered_cols.append(col)
    
    # Reorder dataframe
    final_cols = [col for col in ordered_cols if col in df.columns]
    return df[final_cols]


def create_serialized_asset_view(df, is_glassbeam, filepath_dict):
    """Create serialized asset view with L1/L2/L3 modality columns."""
    cols = ['company_name', 'make_source', 'model_name_source', 'model_number']
    if not is_glassbeam:
        cols.append('asset_sys_id')
    if is_glassbeam:
        cols.append('product_mgmt_stage')
    
    source_df = pd.read_excel(filepath_dict['source'], dtype=str, na_filter=False)
    source_df = source_df.fillna('').astype(str)
    source_df = source_df.apply(lambda x: x.str.strip())
    
    # Map source modality columns
    source_cols = config.SOURCE_COLUMNS
    for level in ['l1', 'l2', 'l3']:
        config_key = f'{level}_modality_source'
        if config_key in source_cols:
            source_col_name = source_cols[config_key]
            if source_col_name in source_df.columns:
                source_df[f'{level}_modality_source'] = source_df[source_col_name]
            else:
                source_df[f'{level}_modality_source'] = ''
        else:
            source_df[f'{level}_modality_source'] = ''
    
    # Add normalized source columns
    source_df = add_normalized_columns_source(source_df)
    source_df['make_source_normalized'] = source_df['make_source'].apply(config.normalize_name)
    
    # Handle missing columns
    if 'model_number' not in source_df.columns:
        source_df['model_number'] = ''
        source_df['model_number_normalized'] = ''
    if 'company_name' not in source_df.columns:
        source_df['company_name'] = ''
    
    # Build column list with modalities
    modality_cols = []
    for level in ['l1', 'l2', 'l3']:
        modality_cols.extend([f'{level}_modality_source', f'{level}_modality_source_normalized'])
    
    cols_with_normalized = cols + ['make_source_normalized', 'model_name_source_normalized', 'model_number_normalized'] + modality_cols
    cols_with_normalized = [c for c in cols_with_normalized if c in source_df.columns]
    source_df = source_df[cols_with_normalized].fillna('').astype(str)
    
    # Standardize make
    make_mapping_df = pd.read_excel(filepath_dict['make_mapping'], dtype=str, na_filter=False)
    make_mapping_df = make_mapping_df.fillna('').astype(str)
    make_mapping_df = make_mapping_df.apply(lambda x: x.str.strip())
    make_mapping_df = make_mapping_df.rename(columns={'match_type': 'make_match_type', 'confidence': 'make_confidence'})
    
    if 'make_source_normalized' in make_mapping_df.columns:
        output_df = pd.merge(source_df, make_mapping_df, on='make_source_normalized', how='left', suffixes=('', '_mapping')).fillna('')
    else:
        output_df = pd.merge(source_df, make_mapping_df, on='make_source', how='left').fillna('')
        if 'make_target' in output_df.columns:
            output_df['make_target_normalized'] = output_df['make_target'].apply(config.normalize_name)
    
    # Merge with model mapping
    if 'make_target_normalized' in df.columns and 'model_name_source_normalized' in df.columns:
        output_df = pd.merge(output_df, df, on=['make_target_normalized', 'model_name_source_normalized'], how='left', suffixes=('', '_model')).fillna('')
    else:
        output_df = pd.merge(output_df, df, on=['make_target', 'model_name_source'], how='left').fillna('')
    
    return output_df


def remove_preexisting_matches(source_df, model_mapping_filepath):
    """Remove preexisting matches using normalized fields.
    If REMAP_UNMATCHED is True, only keep records where model_match_type != 'no_match'.
    """
    if os.path.exists(model_mapping_filepath):
        model_mapping_df = pd.read_excel(model_mapping_filepath, dtype=str, na_filter=False)
        model_mapping_df = model_mapping_df.apply(lambda x: x.str.strip())
        model_mapping_df = model_mapping_df.fillna('').astype(str)
        
        if 'make_target_normalized' in model_mapping_df.columns and 'model_name_source_normalized' in model_mapping_df.columns:
            cols_to_keep = ['make_target_normalized', 'model_name_source_normalized']
            if 'model_match_type' in model_mapping_df.columns:
                cols_to_keep.append('model_match_type')
            model_mapping_df = model_mapping_df[cols_to_keep]
            
            # If REMAP_UNMATCHED, exclude records where model_match_type = 'no_match'
            if config.REMAP_UNMATCHED and 'model_match_type' in model_mapping_df.columns:
                model_mapping_df = model_mapping_df[model_mapping_df['model_match_type'].str.lower() != 'no_match']
            
            model_mapping_df['existing_record'] = True
            source_df = pd.merge(source_df, model_mapping_df, on=['make_target_normalized', 'model_name_source_normalized'], how='left').fillna('')
        else:
            cols_to_keep = ['make_target', 'model_name_source']
            if 'model_match_type' in model_mapping_df.columns:
                cols_to_keep.append('model_match_type')
            model_mapping_df = model_mapping_df[cols_to_keep]
            
            # If REMAP_UNMATCHED, exclude records where model_match_type = 'no_match'
            if config.REMAP_UNMATCHED and 'model_match_type' in model_mapping_df.columns:
                model_mapping_df = model_mapping_df[model_mapping_df['model_match_type'].str.lower() != 'no_match']
            
            model_mapping_df['existing_record'] = True
            source_df = pd.merge(source_df, model_mapping_df, on=['make_target', 'model_name_source'], how='left').fillna('')
        
        source_df = source_df[~(source_df['existing_record'] == True)]
    
    if 'model_name_source_normalized' in source_df.columns:
        source_df = source_df[source_df['model_name_source_normalized'].str.strip() != '']
    else:
        source_df = source_df[source_df['model_name_source'].str.strip() != '']
    
    return source_df


def process_make(make_target_normalized, make_target, source_df, target_df, filepath_dict, i, total_makes, taxonomy):
    """
    Process all models for one manufacturer using hierarchical modality mapping.
    """
    print(f"Processing manufacturer {str(i+1)} of {total_makes}: {make_target} (normalized: {make_target_normalized})")
    
    this_make_target_df = target_df[target_df['make_target_normalized'] == make_target_normalized]
    this_make_source_df = source_df[source_df['make_target_normalized'] == make_target_normalized]
    
    # Map models
    make_df = map_one_make(this_make_source_df, this_make_target_df, make_target_normalized, make_target)
    
    if len(make_df) > 0:
        # Backfill MEL ID and L1/L2/L3 modalities for matched models
        make_df = postprocess_make_df(make_df, this_make_target_df, make_target, make_target_normalized)
        
        # Hierarchical modality mapping for unmatched models
        make_df = map_unmatched_devices(make_df, taxonomy)
        
        # Remove unwanted columns
        cols_to_remove = [x for x in make_df.columns if 'priority' in x or 'Unnamed:' in x or x in ['make_source', 'match_type', 'existing_record']]
        make_df = make_df[[x for x in make_df.columns if x not in cols_to_remove]]
        
        make_df['added_on'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save to batch file
        if not os.path.exists(filepath_dict['batch_folder']):
            os.makedirs(filepath_dict['batch_folder'])
        batch_filepath = filepath_dict['batch_folder'] + f'/{str(i)}.xlsx'
        make_df.to_excel(batch_filepath, index=False)


def print_summary(serialized_df, is_glassbeam, filepath_dict):
    """Print summary statistics."""
    source_df = pd.read_excel(filepath_dict['source'], dtype=str, na_filter=False)
    source_df = source_df.apply(lambda x: x.str.strip())
    
    n_source = len(source_df)
    print(f'N assets (source): {str(n_source)}')
    
    if is_glassbeam == False:
        n_source_deduped = len(source_df.drop_duplicates(subset=['asset_sys_id']))
    else:
        n_source_deduped = len(source_df.drop_duplicates(subset=['make_source', 'model_name_source']))
    print(f'N assets (source, deduped): {str(n_source_deduped)}')
    
    n_serialized = len(serialized_df)
    print(f'N assets (serialized): {str(n_serialized)}')
    
    if is_glassbeam == False:
        n_serialized_deduped = len(serialized_df.drop_duplicates(subset=['asset_sys_id']))
    else:
        n_serialized_deduped = len(serialized_df.drop_duplicates(subset=['make_source', 'model_name_source']))
    print(f'N assets (serialized, deduped): {str(n_serialized_deduped)}')
    
    if 'mdm_model_id' in serialized_df.columns:
        n_w_mdm_id = len(serialized_df[serialized_df['mdm_model_id'].str.strip() != ''])
        print(f'N assets w/ MDM Model ID: {str(n_w_mdm_id)}')
        
        # Check L3 modality for unmatched
        if 'l3_modality_target' in serialized_df.columns:
            n_wo_mdm_with_l3 = len(serialized_df[(serialized_df['mdm_model_id'].str.strip() == '') & (serialized_df['l3_modality_target'].str.strip() != '')])
            print(f'N assets w/o MDM Model ID but with L3 modality: {str(n_wo_mdm_with_l3)}')
            
            n_with_nothing = len(serialized_df[(serialized_df['mdm_model_id'].str.strip() == '') & (serialized_df['l3_modality_target'].str.strip() == '')])
            print(f'N assets w/o MDM Model ID or L3 modality: {str(n_with_nothing)}')


def process_one_source_file(source_rump, is_glassbeam, target_df, taxonomy):
    """
    Process a single source file for model mapping using hierarchical modality mapping.
    """
    global _current_batch_info
    
    filepath_dict = config.get_filepaths(source_rump)
    
    # Set batch info for cleanup handlers
    _current_batch_info['batch_folder'] = filepath_dict['batch_folder']
    _current_batch_info['model_mapping_path'] = filepath_dict['model_mapping']
    _current_batch_info['cleanup_done'] = False
    
    print(f'\n{"="*60}')
    print(f'Processing source file: {source_rump}')
    print(f'{"="*60}')
    
    if not os.path.exists(filepath_dict['source']):
        print(f'Warning: Source file not found: {filepath_dict["source"]}')
        return None
    
    source_df = get_source_data(filepath_dict['source'])
    source_df = standardize_make(source_df, filepath_dict['make_mapping'])
    
    this_target_df = target_df.copy()
    
    source_df = remove_preexisting_matches(source_df, filepath_dict['model_mapping'])
    
    make_pairs = source_df[['make_target', 'make_target_normalized']].drop_duplicates()
    makes_list = sorted(make_pairs['make_target_normalized'].unique(), reverse=True)
    norm_to_original = dict(zip(make_pairs['make_target_normalized'], make_pairs['make_target']))
    
    print(f"Found {len(makes_list)} new makes for {source_rump}")
    
    if len(makes_list) > 0:
        with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures = []
            for i, make_target_normalized in enumerate(makes_list):
                make_target = norm_to_original.get(make_target_normalized, make_target_normalized)
                future = executor.submit(
                    process_make,
                    make_target_normalized,
                    make_target,
                    source_df,
                    this_target_df,
                    filepath_dict,
                    i,
                    len(makes_list),
                    taxonomy
                )
                futures.append(future)
            
            n_completed = 0
            for future in as_completed(futures):
                n_completed += 1
                print(f"[{source_rump}] Completed {n_completed} of {len(makes_list)} makes")
        
        if os.path.exists(filepath_dict['batch_folder']):
            batch_files = [os.path.join(filepath_dict['batch_folder'], file) for file in os.listdir(filepath_dict['batch_folder']) if file.endswith('.xlsx')]
            if len(batch_files) > 0:
                df = join_batch_files(batch_files)
                
                # Remove unwanted columns from model_mapping.csv
                cols_to_remove = ['model_confidence', 'modality_confidence', 
                                  'l1_modality_confidence', 'l2_modality_confidence', 'l3_modality_confidence']
                df = df.drop(columns=[col for col in cols_to_remove if col in df.columns], errors='ignore')
                
                # Reorder columns for model_mapping.csv
                df = reorder_output_columns(df, is_glassbeam)
                
                print(f'\n[{source_rump}] Make/model summary')
                df = config.save_new_file(
                    df,
                    filepath_dict['model_mapping'],
                    append_to_old=True,
                    timeout=config.CONCURRENT_WRITE_TIMEOUT_LONG,
                    unique_cols=['make_target_normalized', 'model_name_source_normalized'],
                    tiebreak_cols=['mel_id', 'l3_modality_target'],
                    print_stats=True
                )
                for file in batch_files:
                    os.remove(file)
                
                # Mark cleanup as done since batch files were successfully saved and removed
                _current_batch_info['cleanup_done'] = True
            else:
                df = pd.read_excel(filepath_dict['model_mapping'], dtype=str, na_filter=False)
                df = df.apply(lambda col: col.str.strip())
        else:
            df = pd.read_excel(filepath_dict['model_mapping'], dtype=str, na_filter=False)
            df = df.apply(lambda col: col.str.strip())
    else:
        if os.path.exists(filepath_dict['model_mapping']):
            df = pd.read_excel(filepath_dict['model_mapping'], dtype=str, na_filter=False)
            df = df.apply(lambda col: col.str.strip())
        else:
            print(f"Warning: No model mapping file found for {source_rump}")
            return None
    
    df = create_serialized_asset_view(df, is_glassbeam, filepath_dict)
    
    if is_glassbeam:
        unique_cols = ['make_source', 'model_name_source']
        tiebreak_cols = ['mdm_model_id', 'l3_modality_target']
    else:
        unique_cols = ['company_name', 'asset_sys_id']
        tiebreak_cols = ['mdm_model_id', 'l3_modality_target']
    
    df = df[[col for col in df.columns if 'priority' not in col and 'Unnamed:' not in col and 'existing_record' not in col]]
    
    # Rename mel_id to mdm_model_id for output
    if 'mel_id' in df.columns:
        df = df.rename(columns={'mel_id': 'mdm_model_id'})
    
    # Clean serialized dataframe (remove unwanted cols, uppercase/trim identifiers)
    df = clean_serialized_df(df, is_glassbeam)
    
    # Reorder columns for better readability
    df = reorder_output_columns(df, is_glassbeam)
    
    df = config.save_new_file(df, filepath_dict['serialized_asset_view'], append_to_old=False, unique_cols=unique_cols, tiebreak_cols=tiebreak_cols)
    print_summary(df, is_glassbeam, filepath_dict)
    
    # Run validation tests
    run_validation_tests(df, is_glassbeam, filepath_dict, target_df)
    
    # Append to consolidated all-customers file (df is already cleaned and reordered)
    append_to_all_customers_file(df, is_glassbeam)
    
    return df


if __name__ == "__main__":
    source_rumps = config.get_source_rump()
    
    if not source_rumps:
        print('No source files to process.')
        exit()
    
    if isinstance(source_rumps, str):
        source_rumps = [source_rumps]
    
    is_glassbeam_str = input('Are these glassbeam files? (y/n): ')
    is_glassbeam = True if is_glassbeam_str.lower().strip() == 'y' else False
    
    print(f'\nProcessing {len(source_rumps)} source file(s)...')
    
    # Pre-load target data and build taxonomy
    first_filepath_dict = config.get_filepaths(source_rumps[0])
    target_df = get_target_data(first_filepath_dict)
    taxonomy = build_taxonomy_lookup(target_df)
    
    print(f"\nTaxonomy built: {len(taxonomy['l1_list'])} L1, "
          f"{sum(len(v) for v in taxonomy['l1_to_l2'].values())} L2, "
          f"{sum(len(v) for v in taxonomy['l1_l2_to_l3'].values())} L3 categories")
    
    for source_rump in source_rumps:
        process_one_source_file(source_rump, is_glassbeam, target_df, taxonomy)
    
    print(f'\n{"="*60}')
    print(f'COMPLETE: Processed {len(source_rumps)} source file(s)')
    print(f'{"="*60}')
