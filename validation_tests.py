"""
Validation tests for asset standardization output files.
"""

import pandas as pd
import os


def run_validation_tests(serialized_df, is_glassbeam, filepath_dict, target_df):
    """
    Run validation tests on the output files and print results.
    """
    print(f'\n{"-"*60}')
    print('VALIDATION TESTS')
    print(f'{"-"*60}')
    
    all_passed = True
    
    # ========================================================================
    # TEST 1: L1/L2/L3 combinations should exist in MEL
    # ========================================================================
    print('\n[TEST 1] L1/L2/L3 combinations validation:')
    
    # Build MEL taxonomy lookups at each level
    mel_l1_set = set()  # Valid L1 values
    mel_l1_l2_set = set()  # Valid (L1, L2) combinations
    mel_l1_l2_l3_set = set()  # Valid (L1, L2, L3) combinations
    
    for _, row in target_df.iterrows():
        l1 = str(row.get('l1_modality_target', '')).strip().upper()
        l2 = str(row.get('l2_modality_target', '')).strip().upper()
        l3 = str(row.get('l3_modality_target', '')).strip().upper()
        
        if l1:
            mel_l1_set.add(l1)
        if l1 and l2:
            mel_l1_l2_set.add((l1, l2))
        if l1 and l2 and l3:
            mel_l1_l2_l3_set.add((l1, l2, l3))
    
    # Check serialized asset view combinations hierarchically
    invalid_combos = []
    for _, row in serialized_df.iterrows():
        l1 = str(row.get('l1_modality_target', '')).strip().upper()
        l2 = str(row.get('l2_modality_target', '')).strip().upper()
        l3 = str(row.get('l3_modality_target', '')).strip().upper()
        
        is_valid = True
        
        # If L3 is set, check full (L1, L2, L3) combination
        if l3:
            if (l1, l2, l3) not in mel_l1_l2_l3_set:
                is_valid = False
        # If L2 is set but L3 is blank, check (L1, L2) combination
        elif l2:
            if (l1, l2) not in mel_l1_l2_set:
                is_valid = False
        # If only L1 is set, check L1 exists
        elif l1:
            if l1 not in mel_l1_set:
                is_valid = False
        
        if not is_valid:
            invalid_combos.append((l1, l2, l3))
    
    invalid_combos_unique = list(set(invalid_combos))
    if len(invalid_combos_unique) == 0:
        print('  ✓ PASSED: All L1/L2/L3 combinations exist in MEL')
    else:
        print(f'  ✗ FAILED: {len(invalid_combos_unique)} invalid L1/L2/L3 combinations found:')
        for combo in invalid_combos_unique[:10]:  # Show first 10
            print(f'    - L1: {combo[0]}, L2: {combo[1]}, L3: {combo[2]}')
        if len(invalid_combos_unique) > 10:
            print(f'    ... and {len(invalid_combos_unique) - 10} more')
        all_passed = False
    
    # ========================================================================
    # TEST 2: Match percentages
    # ========================================================================
    print('\n[TEST 2] Match percentages:')
    total_assets = len(serialized_df)
    
    if total_assets > 0:
        # Make match
        if 'make_match_type' in serialized_df.columns:
            make_matched = len(serialized_df[serialized_df['make_match_type'].str.lower() != 'no_match'])
            print(f'  Make match: {make_matched}/{total_assets} ({100*make_matched/total_assets:.1f}%)')
        
        # Model match
        if 'model_match_type' in serialized_df.columns:
            model_matched = len(serialized_df[serialized_df['model_match_type'].str.lower() != 'no_match'])
            print(f'  Model match: {model_matched}/{total_assets} ({100*model_matched/total_assets:.1f}%)')
        
        # Modality matches (L1, L2, L3)
        for level in ['l1', 'l2', 'l3']:
            col = f'{level}_modality_target'
            if col in serialized_df.columns:
                modality_matched = len(serialized_df[serialized_df[col].str.strip() != ''])
                print(f'  {level.upper()} modality: {modality_matched}/{total_assets} ({100*modality_matched/total_assets:.1f}%)')
    
    # ========================================================================
    # TEST 3: No duplicates in serialized_asset_view
    # ========================================================================
    print('\n[TEST 3] Duplicates in serialized_asset_view:')
    
    if not is_glassbeam:
        if 'company_name' in serialized_df.columns and 'asset_sys_id' in serialized_df.columns:
            serialized_df['_test_key'] = (serialized_df['company_name'].str.strip().str.upper() + '|' + 
                                          serialized_df['asset_sys_id'].str.strip().str.upper())
            n_dupes = len(serialized_df) - len(serialized_df['_test_key'].unique())
            serialized_df = serialized_df.drop(columns=['_test_key'])
            
            if n_dupes == 0:
                print('  ✓ PASSED: No duplicates found (by company_name + asset_sys_id)')
            else:
                print(f'  ✗ FAILED: {n_dupes} duplicate records found')
                all_passed = False
        else:
            print('  ⚠ SKIPPED: company_name or asset_sys_id column missing')
    else:
        if 'make_source' in serialized_df.columns and 'model_name_source' in serialized_df.columns:
            serialized_df['_test_key'] = (serialized_df['make_source'].str.strip().str.upper() + '|' + 
                                          serialized_df['model_name_source'].str.strip().str.upper())
            n_dupes = len(serialized_df) - len(serialized_df['_test_key'].unique())
            serialized_df = serialized_df.drop(columns=['_test_key'])
            
            if n_dupes == 0:
                print('  ✓ PASSED: No duplicates found (by make_source + model_name_source)')
            else:
                print(f'  ✗ FAILED: {n_dupes} duplicate records found')
                all_passed = False
    
    # ========================================================================
    # TEST 4: Conflicting model mappings in model_mapping.csv
    # Check if same (make_target + model_name_source_normalized) maps to different model_name_target values
    # ========================================================================
    print('\n[TEST 4] Conflicting model mappings in model_mapping.csv:')
    
    if os.path.exists(filepath_dict['model_mapping']):
        model_mapping_df = pd.read_excel(filepath_dict['model_mapping'], dtype=str, na_filter=False)
        model_mapping_df = model_mapping_df.fillna('').astype(str)
        
        required_cols = ['make_target', 'model_name_source_normalized', 'model_name_target']
        if all(col in model_mapping_df.columns for col in required_cols):
            # Create key from make_target + model_name_source_normalized
            model_mapping_df['_source_key'] = (model_mapping_df['make_target'].str.strip().str.upper() + '|' + 
                                                model_mapping_df['model_name_source_normalized'].str.strip().str.upper())
            model_mapping_df['_target_normalized'] = model_mapping_df['model_name_target'].str.strip().str.upper()
            
            # Group by source key and count unique targets
            conflicts = model_mapping_df.groupby('_source_key')['_target_normalized'].nunique()
            conflicting_keys = conflicts[conflicts > 1]
            
            if len(conflicting_keys) == 0:
                print('  ✓ PASSED: No conflicting model mappings found')
            else:
                print(f'  ✗ FAILED: {len(conflicting_keys)} source models map to multiple different targets:')
                for key in list(conflicting_keys.index)[:5]:
                    targets = model_mapping_df[model_mapping_df['_source_key'] == key]['_target_normalized'].unique()
                    print(f'    - {key} -> {list(targets)}')
                if len(conflicting_keys) > 5:
                    print(f'    ... and {len(conflicting_keys) - 5} more')
                all_passed = False
        else:
            missing = [col for col in required_cols if col not in model_mapping_df.columns]
            print(f'  ⚠ SKIPPED: Missing columns: {missing}')
    else:
        print('  ⚠ SKIPPED: model_mapping.csv not found')
    
    # ========================================================================
    # TEST 5: Conflicting make mappings in make_mapping.csv
    # Check if same make_source_normalized maps to different make_target values
    # ========================================================================
    print('\n[TEST 5] Conflicting make mappings in make_mapping.csv:')
    
    if os.path.exists(filepath_dict['make_mapping']):
        make_mapping_df = pd.read_excel(filepath_dict['make_mapping'], dtype=str, na_filter=False)
        make_mapping_df = make_mapping_df.fillna('').astype(str)
        
        required_cols = ['make_source_normalized', 'make_target']
        if all(col in make_mapping_df.columns for col in required_cols):
            # Normalize for comparison
            make_mapping_df['_source_key'] = make_mapping_df['make_source_normalized'].str.strip().str.upper()
            make_mapping_df['_target_normalized'] = make_mapping_df['make_target'].str.strip().str.upper()
            
            # Group by source key and count unique targets
            conflicts = make_mapping_df.groupby('_source_key')['_target_normalized'].nunique()
            conflicting_keys = conflicts[conflicts > 1]
            
            if len(conflicting_keys) == 0:
                print('  ✓ PASSED: No conflicting make mappings found')
            else:
                print(f'  ✗ FAILED: {len(conflicting_keys)} source makes map to multiple different targets:')
                for key in list(conflicting_keys.index)[:5]:
                    targets = make_mapping_df[make_mapping_df['_source_key'] == key]['_target_normalized'].unique()
                    print(f'    - {key} -> {list(targets)}')
                if len(conflicting_keys) > 5:
                    print(f'    ... and {len(conflicting_keys) - 5} more')
                all_passed = False
        else:
            missing = [col for col in required_cols if col not in make_mapping_df.columns]
            print(f'  ⚠ SKIPPED: Missing columns: {missing}')
    else:
        print('  ⚠ SKIPPED: make_mapping.csv not found')
    
    # ========================================================================
    # TEST 6: Source asset count = Serialized asset count
    # ========================================================================
    print('\n[TEST 6] Asset count validation:')
    
    if is_glassbeam:
        print('  ⚠ SKIPPED: Not applicable to glassbeam files, since they are not at the individual asset level')
    else:
        source_df = pd.read_excel(filepath_dict['source'], dtype=str, na_filter=False)
        source_df = source_df.fillna('').astype(str)
        source_df = source_df.apply(lambda x: x.str.strip())
        
        if 'asset_sys_id' not in source_df.columns:
            print('  ✗ FAILED: asset_sys_id column missing in source file')
            all_passed = False
        elif 'asset_sys_id' not in serialized_df.columns:
            print('  ✗ FAILED: asset_sys_id column missing in serialized file')
            all_passed = False
        else:
            source_df['_test_key'] = source_df['asset_sys_id'].str.strip().str.upper()
            n_source_unique = len(source_df['_test_key'].unique())
            
            serialized_df_copy = serialized_df.copy()
            serialized_df_copy['_test_key'] = serialized_df_copy['asset_sys_id'].str.strip().str.upper()
            n_serialized_unique = len(serialized_df_copy['_test_key'].unique())
            
            if n_source_unique == n_serialized_unique:
                print(f'  ✓ PASSED: Source assets ({n_source_unique}) = Serialized assets ({n_serialized_unique})')
            else:
                print(f'  ✗ FAILED: Source assets ({n_source_unique}) != Serialized assets ({n_serialized_unique})')
                all_passed = False
    
    # ========================================================================
    # TEST 7: No blanks in make_target or model_name_target
    # ========================================================================
    print('\n[TEST 7] Blank target values:')
    
    # Check make_mapping.csv
    if os.path.exists(filepath_dict['make_mapping']):
        make_mapping_df = pd.read_excel(filepath_dict['make_mapping'], dtype=str, na_filter=False)
        make_mapping_df = make_mapping_df.fillna('').astype(str)
        
        if 'make_target' in make_mapping_df.columns:
            blank_make_targets = len(make_mapping_df[make_mapping_df['make_target'].str.strip() == ''])
            if blank_make_targets == 0:
                print('  ✓ PASSED: No blanks in make_mapping.make_target')
            else:
                print(f'  ✗ FAILED: {blank_make_targets} blank values in make_mapping.make_target')
                all_passed = False
    
    # Check model_mapping.csv
    if os.path.exists(filepath_dict['model_mapping']):
        model_mapping_df = pd.read_excel(filepath_dict['model_mapping'], dtype=str, na_filter=False)
        model_mapping_df = model_mapping_df.fillna('').astype(str)
        
        if 'model_name_target' in model_mapping_df.columns:
            blank_model_targets = len(model_mapping_df[model_mapping_df['model_name_target'].str.strip() == ''])
            if blank_model_targets == 0:
                print('  ✓ PASSED: No blanks in model_mapping.model_name_target')
            else:
                print(f'  ✗ FAILED: {blank_model_targets} blank values in model_mapping.model_name_target')
                all_passed = False
    
    # ========================================================================
    # TEST 8: verified_model_name consistency
    # ========================================================================
    print('\n[TEST 8] verified_model_name consistency:')
    
    if 'verified_model_name' in serialized_df.columns and 'mel_id' in serialized_df.columns and 'model_match_type' in serialized_df.columns:
        # Convert verified_model_name to string for comparison
        serialized_df_test = serialized_df.copy()
        serialized_df_test['verified_model_name'] = serialized_df_test['verified_model_name'].astype(str).str.strip()
        serialized_df_test['mel_id'] = serialized_df_test['mel_id'].astype(str).str.strip()
        serialized_df_test['model_match_type'] = serialized_df_test['model_match_type'].astype(str).str.strip().str.lower()
        
        # Check: verified_model_name = 0 should have mel_id blank and model_match_type = 'no_match'
        unverified_mask = serialized_df_test['verified_model_name'] == '0'
        unverified_with_mel = serialized_df_test[unverified_mask & (serialized_df_test['mel_id'] != '')]
        unverified_not_no_match = serialized_df_test[unverified_mask & (serialized_df_test['model_match_type'] != 'no_match')]
        
        # Check: verified_model_name = 1 should have mel_id non-blank and model_match_type != 'no_match'
        verified_mask = serialized_df_test['verified_model_name'] == '1'
        verified_no_mel = serialized_df_test[verified_mask & (serialized_df_test['mel_id'] == '')]
        verified_is_no_match = serialized_df_test[verified_mask & (serialized_df_test['model_match_type'] == 'no_match')]
        
        test8_passed = True
        
        if len(unverified_with_mel) > 0:
            print(f'  ✗ FAILED: {len(unverified_with_mel)} rows with verified_model_name=0 have non-blank mel_id')
            test8_passed = False
            all_passed = False
        
        if len(unverified_not_no_match) > 0:
            print(f'  ✗ FAILED: {len(unverified_not_no_match)} rows with verified_model_name=0 have model_match_type != "no_match"')
            test8_passed = False
            all_passed = False
        
        if len(verified_no_mel) > 0:
            print(f'  ⚠ WARNING: {len(verified_no_mel)} rows with verified_model_name=1 have blank mel_id (LLM matches may not have MEL ID)')
            # Note: This is a warning, not a failure, because LLM matches might not have mel_id
        
        if len(verified_is_no_match) > 0:
            print(f'  ✗ FAILED: {len(verified_is_no_match)} rows with verified_model_name=1 have model_match_type = "no_match"')
            test8_passed = False
            all_passed = False
        
        if test8_passed:
            print('  ✓ PASSED: verified_model_name is consistent with mel_id and model_match_type')
    else:
        print('  ⚠ SKIPPED: Required columns missing (verified_model_name, mel_id, or model_match_type)')
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f'\n{"-"*60}')
    if all_passed:
        print('ALL VALIDATION TESTS PASSED ✓')
    else:
        print('SOME VALIDATION TESTS FAILED ✗')
    print(f'{"-"*60}')
    
    return all_passed
