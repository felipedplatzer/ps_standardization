"""
Supra-script to run make_mapping and model_mapping together.
Handles all user inputs and orchestrates the full pipeline.
"""

import config
import make_mapping
import model_mapping
import os


def get_user_inputs():
    """
    Get all user inputs needed for the pipeline.
    Returns: (source_rumps, is_glassbeam)
    """
    # Get source file names (can be multiple or all files)
    source_rumps = config.get_source_rump()
    
    if not source_rumps:
        print('No source files to process.')
        return None, None
    
    if isinstance(source_rumps, str):
        source_rumps = [source_rumps]
    
    # Check if glassbeam files
    is_glassbeam_str = input('Are these glassbeam files? (y/n): ')
    is_glassbeam = True if is_glassbeam_str.lower().strip() == 'y' else False
    
    return source_rumps, is_glassbeam


def run_make_mapping(source_rumps):
    """
    Run make mapping for all source files.
    """
    print('\n' + '='*60)
    print('STEP 1: MAKE MAPPING')
    print('='*60)
    
    total_new_mappings = 0
    for source_rump in source_rumps:
        filepath_dict = config.get_filepaths(source_rump)
        
        # Check if source file exists
        if not os.path.exists(filepath_dict['source']):
            print(f'\nWarning: Source file not found: {filepath_dict["source"]}')
            continue
        
        n_mappings = make_mapping.process_one_source_file(source_rump, filepath_dict)
        total_new_mappings += n_mappings
    
    print(f'\nMake mapping complete: {total_new_mappings} new mappings created')
    return total_new_mappings


def run_model_mapping(source_rumps, is_glassbeam):
    """
    Run model mapping for all source files.
    """
    print('\n' + '='*60)
    print('STEP 2: MODEL MAPPING')
    print('='*60)
    
    # Pre-load target data and build taxonomy
    first_filepath_dict = config.get_filepaths(source_rumps[0])
    target_df = model_mapping.get_target_data(first_filepath_dict)
    taxonomy = model_mapping.build_taxonomy_lookup(target_df)
    
    print(f"\nTaxonomy built: {len(taxonomy['l1_list'])} L1, "
          f"{sum(len(v) for v in taxonomy['l1_to_l2'].values())} L2, "
          f"{sum(len(v) for v in taxonomy['l1_l2_to_l3'].values())} L3 categories")
    
    for source_rump in source_rumps:
        model_mapping.process_one_source_file(source_rump, is_glassbeam, target_df, taxonomy)
    
    print(f'\nModel mapping complete for {len(source_rumps)} file(s)')


def main():
    """
    Main entry point for the full pipeline.
    """
    print('='*60)
    print('ASSET STANDARDIZATION PIPELINE')
    print('='*60)
    
    # Get user inputs
    source_rumps, is_glassbeam = get_user_inputs()
    
    if source_rumps is None:
        return
    
    print(f'\nProcessing {len(source_rumps)} source file(s)...')
    print(f'Glassbeam mode: {is_glassbeam}')
    
    # Step 1: Make mapping
    run_make_mapping(source_rumps)
    
    # Step 2: Model mapping
    run_model_mapping(source_rumps, is_glassbeam)
    
    # Final summary
    print('\n' + '='*60)
    print('PIPELINE COMPLETE')
    print('='*60)
    print(f'Processed {len(source_rumps)} source file(s)')
    print('Output files:')
    for source_rump in source_rumps:
        filepath_dict = config.get_filepaths(source_rump)
        print(f'  - {filepath_dict["serialized_asset_view"]}')
    print(f'  - {model_mapping.get_all_customers_filepath()}')


if __name__ == "__main__":
    main()
