import pandas as pd
import config
import llm_modality_mapper


def split_df(source_df):
    unmatched_df = source_df[source_df['model_match_type'].str.lower() == 'no_match']
    matched_df = source_df[source_df['model_match_type'].str.lower() != 'no_match']
    matched_df['modality_match_type'] = 'from_mel_based_on_model'
    unmatched_df = unmatched_df.sort_values(by=['make_target','model_name_source'])
    return unmatched_df, matched_df

def modality_mapping(source_df, target_modality_list):
    # Split df into matched and unmatched
    unmatched_df, matched_df = split_df(source_df)
    if len(unmatched_df) == 0:
        return matched_df
    else:
        print('FILLING OUT MODALITIES')
        # Convert to dict list for LLM processing (this is necessary since the LLM function expects dict list)
        unmatched_dl = unmatched_df.to_dict(orient='records')
        # Classify devices
        unmatched_dl_processed = llm_modality_mapper.classify_devices_from_dataframe(unmatched_dl, target_modality_list)    
        # Convert back to DataFrame and merge with original unmatched data to preserve all columns
        unmatched_processed_df = pd.DataFrame(unmatched_dl_processed)
        unmatched_processed_df['modality_match_type'] = 'llm_based_on_make_and_model'
        # Merge the classification results back with the original unmatched data
        # Use make_target and model_name_source as merge keys
        x = unmatched_df.drop(columns=['modality_target', 'modality_confidence']) # target and confidence come from processed df (i..e after passing through the LLM) 
        unmatched_df_merged = pd.merge(
            x, 
            unmatched_processed_df[['make_target', 'model_name_source', 'modality_target', 'modality_match_type', 'modality_confidence']], 
            on=['make_target', 'model_name_source'], 
            how='left'
        )
        # Update modality_target and modality_confidence for unmatched rows
        unmatched_df_merged['modality_target'] = unmatched_df_merged['modality_target'].fillna('')
        unmatched_df_merged['modality_confidence'] = unmatched_df_merged['modality_confidence'].fillna(0)
        # Merge with matched df
        output_df = pd.concat([matched_df, unmatched_df_merged])
        return output_df










