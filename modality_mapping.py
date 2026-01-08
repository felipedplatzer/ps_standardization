import pandas as pd
import config
import llm_modality_mapper


def split_df(source_df):
    unmatched_df = source_df[source_df['model_match_type'].str.lower() == 'no_match']
    matched_df = source_df[source_df['model_match_type'].str.lower() != 'no_match']
    matched_df['modality_match_type'] = 'from_mel_based_on_model'
    # Sort by normalized fields if available, otherwise use raw fields
    if 'make_target_normalized' in unmatched_df.columns and 'model_name_source_normalized' in unmatched_df.columns:
        unmatched_df = unmatched_df.sort_values(by=['make_target_normalized', 'model_name_source_normalized'])
    else:
        unmatched_df = unmatched_df.sort_values(by=['make_target', 'model_name_source'])
    return unmatched_df, matched_df


def modality_mapping(source_df, target_modality_list):
    # Split df into matched and unmatched
    unmatched_df, matched_df = split_df(source_df)
    if len(unmatched_df) == 0:
        return matched_df
    else:
        print('FILLING OUT MODALITIES')
        # Convert to dict list for LLM processing
        unmatched_dl = unmatched_df.to_dict(orient='records')
        # Classify devices
        unmatched_dl_processed = llm_modality_mapper.classify_devices_from_dataframe(unmatched_dl, target_modality_list)
        # Convert back to DataFrame
        unmatched_processed_df = pd.DataFrame(unmatched_dl_processed)
        unmatched_processed_df['modality_match_type'] = 'llm_based_on_make_and_model'
        
        # Add normalized modality_target if not present
        if 'modality_target' in unmatched_processed_df.columns and 'modality_target_normalized' not in unmatched_processed_df.columns:
            unmatched_processed_df['modality_target_normalized'] = unmatched_processed_df['modality_target'].apply(config.normalize_name)
        
        # Determine merge keys based on available columns
        if 'make_target_normalized' in unmatched_df.columns and 'model_name_source_normalized' in unmatched_df.columns:
            merge_keys = ['make_target_normalized', 'model_name_source_normalized']
        else:
            merge_keys = ['make_target', 'model_name_source']
        
        # Columns to drop from original (they come from processed df)
        cols_to_drop = ['modality_target', 'modality_confidence']
        cols_to_drop = [c for c in cols_to_drop if c in unmatched_df.columns]
        if 'modality_target_normalized' in unmatched_df.columns:
            cols_to_drop.append('modality_target_normalized')
        
        x = unmatched_df.drop(columns=cols_to_drop)
        
        # Select columns from processed df for merging
        processed_cols = merge_keys + ['modality_target', 'modality_match_type', 'modality_confidence']
        if 'modality_target_normalized' in unmatched_processed_df.columns:
            processed_cols.append('modality_target_normalized')
        processed_cols = [c for c in processed_cols if c in unmatched_processed_df.columns]
        
        unmatched_df_merged = pd.merge(
            x,
            unmatched_processed_df[processed_cols],
            on=merge_keys,
            how='left'
        ).fillna('')
        
        # Update modality_target and modality_confidence for unmatched rows
        unmatched_df_merged['modality_target'] = unmatched_df_merged['modality_target'].fillna('')
        unmatched_df_merged['modality_confidence'] = unmatched_df_merged['modality_confidence'].fillna(0)
        if 'modality_target_normalized' not in unmatched_df_merged.columns:
            unmatched_df_merged['modality_target_normalized'] = unmatched_df_merged['modality_target'].apply(config.normalize_name)
        
        # Merge with matched df
        output_df = pd.concat([matched_df, unmatched_df_merged])
        return output_df










