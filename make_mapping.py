import config
import match_functions
import pandas as pd
import os
from llm_make_mapper import llm_make_mapper


def get_source_data(source_filepath):
    """Get unique make_source values from source file."""
    df = pd.read_excel(source_filepath, dtype=str, na_filter=False)
    df['make_source'] = df['make_source'].fillna('')
    df = list(df['make_source'].astype(str).unique())
    return sorted(df)


def get_mel_manufacturers(mel_path):
    """
    Get manufacturers and their aliases from MEL file.
    De-duplicates by manufacturer name, keeping first non-empty alias.
    
    Returns:
        DataFrame with columns: make_target, make_target_normalized, manufacturer_aliases
    """
    mel_col = config.MEL_COLUMNS['make_target']
    alias_col = config.MEL_COLUMNS.get('manufacturer_aliases', None)
    
    df = pd.read_excel(mel_path, dtype=str, na_filter=False)
    df = df.fillna('').astype(str)
    df = df.apply(lambda x: x.str.strip())
    
    # Select relevant columns
    cols_to_select = [mel_col]
    if alias_col and alias_col in df.columns:
        cols_to_select.append(alias_col)
        df = df[cols_to_select]
        df = df.rename(columns={mel_col: 'make_target', alias_col: 'manufacturer_aliases'})
    else:
        df = df[[mel_col]]
        df = df.rename(columns={mel_col: 'make_target'})
        df['manufacturer_aliases'] = ''
    
    # Add normalized column
    df['make_target_normalized'] = df['make_target'].apply(config.normalize_name)
    
    # De-duplicate: keep first non-empty alias for each manufacturer
    df['alias_priority'] = df['manufacturer_aliases'].apply(lambda x: 0 if str(x).strip() == '' else 1)
    df = df.sort_values(by='alias_priority', ascending=False)
    df = df.drop_duplicates(subset=['make_target_normalized'], keep='first')
    df = df.drop(columns=['alias_priority'])
    
    # Remove empty manufacturers
    df = df[df['make_target_normalized'].str.strip() != '']
    
    return df


def build_alias_lookup(mel_df):
    """
    Build a lookup from normalized aliases to make_target.
    Splits aliases by '|' and creates entries for each.
    
    Returns:
        dict: normalized_alias -> (make_target, make_target_normalized)
    """
    alias_lookup = {}
    
    for _, row in mel_df.iterrows():
        make_target = row['make_target']
        make_target_normalized = row['make_target_normalized']
        aliases_str = str(row.get('manufacturer_aliases', '')).strip()
        
        if aliases_str:
            # Split by '|' and process each alias
            aliases = [a.strip() for a in aliases_str.split('|') if a.strip()]
            for alias in aliases:
                alias_normalized = config.normalize_name(alias)
                if alias_normalized and alias_normalized not in alias_lookup:
                    alias_lookup[alias_normalized] = {
                        'make_target': make_target,
                        'make_target_normalized': make_target_normalized,
                        'matched_alias': alias
                    }
    
    return alias_lookup


def map_to_mel(source_names_normalized, mel_df):
    """
    Map normalized source names to normalized target names from MEL.
    Uses the manufacturer names (not aliases) for initial mapping.
    """
    target_names_normalized = sorted(mel_df['make_target_normalized'].unique().tolist())
    target_names_normalized = [x for x in target_names_normalized if x.strip() != '']
    
    # Get matches using normalized names
    dl = match_functions.get_all_matches(target_names_normalized, source_names_normalized)
    return dl


def map_to_aliases(unmatched_source_normalized, alias_lookup):
    """
    Try to match unmatched source names against manufacturer aliases.
    
    Args:
        unmatched_source_normalized: List of normalized source names that didn't match
        alias_lookup: Dict from build_alias_lookup()
    
    Returns:
        List of dicts with matches found via aliases
    """
    # Get all normalized aliases as the target list
    alias_list = sorted(list(alias_lookup.keys()))
    
    if not alias_list or not unmatched_source_normalized:
        return []
    
    print(f'Attempting to match {len(unmatched_source_normalized)} unmatched makes against {len(alias_list)} aliases')
    
    # Use the same matching logic as for manufacturer names
    dl = match_functions.get_all_matches(alias_list, unmatched_source_normalized)
    
    # Convert alias matches to make_target matches
    matched_via_alias = []
    for match in dl:
        if match.get('match_type', '') != 'no_match' and match.get('make_target', ''):
            matched_alias_normalized = match['make_target']  # This is the alias that matched
            source_normalized = match['make_source']
            
            if matched_alias_normalized in alias_lookup:
                alias_info = alias_lookup[matched_alias_normalized]
                matched_via_alias.append({
                    'make_source': source_normalized,  # normalized source
                    'make_target': alias_info['make_target_normalized'],  # normalized target
                    'match_type': f"alias_{match.get('match_type', '')}",
                    'matched_alias': alias_info['matched_alias']
                })
    
    print(f'Found {len(matched_via_alias)} matches via aliases')
    return matched_via_alias


def remove_preexisting_matches(source_names_normalized, make_mapping_filepath):
    """
    Remove source names that already have matches in the mapping file.
    """
    if os.path.exists(make_mapping_filepath):
        make_mapping_df = pd.read_excel(make_mapping_filepath, dtype=str, na_filter=False)
        if config.REMAP_UNMATCHED:
            make_mapping_df = make_mapping_df[
                make_mapping_df['make_target_normalized'].notna() & 
                (make_mapping_df['make_target_normalized'].str.strip() != '')
            ]
        preexisting_normalized = list(make_mapping_df['make_source_normalized'].unique())
        new_source_names = [d for d in source_names_normalized if d not in preexisting_normalized]
    else:
        new_source_names = source_names_normalized
    return new_source_names


def create_normalized_lookup(source_names):
    """
    Create a lookup dictionary from normalized names to original names.
    Returns: dict mapping normalized_name -> [list of original names]
    """
    lookup = {}
    for name in source_names:
        normalized = config.normalize_name(name)
        if normalized not in lookup:
            lookup[normalized] = []
        if name not in lookup[normalized]:
            lookup[normalized].append(name)
    return lookup


def process_one_source_file(source_rump, filepath_dict):
    """
    Process a single source file for make mapping with alias support.
    """
    print(f'\n{"="*60}')
    print(f'Processing source file: {source_rump}')
    print(f'{"="*60}')
    
    # Get source data (original names)
    source_names = get_source_data(filepath_dict['source'])
    
    # Get MEL manufacturers with aliases
    mel_df = get_mel_manufacturers(filepath_dict['mel'])
    print(f'Loaded {len(mel_df)} unique manufacturers from MEL')
    
    # Build alias lookup
    alias_lookup = build_alias_lookup(mel_df)
    print(f'Built alias lookup with {len(alias_lookup)} aliases')
    
    # Create lookups for normalized <-> original name mappings
    source_lookup = create_normalized_lookup(source_names)
    
    # Create target lookup (normalized -> original make_target)
    target_lookup = dict(zip(mel_df['make_target_normalized'], mel_df['make_target']))
    
    # Create alias crosswalk (make_target_normalized -> manufacturer_aliases)
    alias_crosswalk = dict(zip(mel_df['make_target_normalized'], mel_df['manufacturer_aliases']))
    
    # Get unique normalized source names
    source_names_normalized = sorted(list(source_lookup.keys()))
    source_names_normalized = [x for x in source_names_normalized if x.strip() != '']
    
    # Remove preexisting matches
    new_source_names_normalized = remove_preexisting_matches(source_names_normalized, filepath_dict['make_mapping'])
    n_preexisting_matches = len(source_names_normalized) - len(new_source_names_normalized)
    
    print(f'Total unique normalized makes in source file: {len(source_names_normalized)}')
    print(f'Preexisting matches: {n_preexisting_matches}')
    print(f'New makes to process: {len(new_source_names_normalized)}')
    
    if len(new_source_names_normalized) == 0:
        print('No new makes to process.')
        return 0
    
    # STEP 1: Map to MEL manufacturer names
    print('\nStep 1: Mapping to MEL manufacturer names...')
    dl = map_to_mel(new_source_names_normalized, mel_df)
    
    # Separate matched and unmatched
    matched_dl = [x for x in dl if x.get('match_type', '') != 'no_match' and x.get('make_target', '')]
    unmatched_dl = [x for x in dl if x.get('match_type', '') == 'no_match' or not x.get('make_target', '')]
    
    print(f'Matched via manufacturer names: {len(matched_dl)}')
    print(f'Unmatched: {len(unmatched_dl)}')
    
    # STEP 2: Try to match unmatched against aliases
    if len(unmatched_dl) > 0:
        print('\nStep 2: Matching unmatched makes against manufacturer aliases...')
        unmatched_source_normalized = [x['make_source'] for x in unmatched_dl]
        alias_matches = map_to_aliases(unmatched_source_normalized, alias_lookup)
        
        # Add alias matches to matched list
        matched_dl.extend(alias_matches)
        
        # Update unmatched list (remove those that matched via alias)
        matched_via_alias_sources = set(x['make_source'] for x in alias_matches)
        unmatched_dl = [x for x in unmatched_dl if x['make_source'] not in matched_via_alias_sources]
        
        print(f'Remaining unmatched after alias matching: {len(unmatched_dl)}')
    
    # Combine all results
    all_results = matched_dl + unmatched_dl
    
    if len(all_results) > 0:
        # Build output rows
        output_rows = []
        for match in all_results:
            source_normalized = match['make_source']
            target_normalized = match.get('make_target', '')
            match_type = match.get('match_type', '')
            matched_alias = match.get('matched_alias', '')
            
            # Get original source names
            original_sources = source_lookup.get(source_normalized, [source_normalized])
            
            # Get original target name
            # If no match found, use source value as target
            if target_normalized:
                original_target = target_lookup.get(target_normalized, target_normalized)
                final_target_normalized = target_normalized
            else:
                # No match - fill with source value
                original_target = original_sources[0] if original_sources else source_normalized
                final_target_normalized = source_normalized
            
            # Get manufacturer aliases for this target (backfill from MEL)
            manufacturer_aliases = alias_crosswalk.get(target_normalized, '') if target_normalized else ''
            
            # Create a row for each original source name
            for original_source in original_sources:
                output_rows.append({
                    'make_source': original_source,
                    'make_source_normalized': source_normalized,
                    'make_target': original_target if target_normalized else original_source,  # Use source if no match
                    'make_target_normalized': final_target_normalized if target_normalized else config.normalize_name(original_source),
                    'manufacturer_aliases': manufacturer_aliases,
                    'match_type': match_type if match_type else 'no_match_use_source',
                    'matched_via_alias': matched_alias
                })
        
        df = pd.DataFrame(output_rows)
        df = df.drop_duplicates(subset=['make_source'])
        
        # Print summary
        n_matched = len(df[df['make_target'].str.strip() != ''])
        n_unmatched = len(df[df['make_target'].str.strip() == ''])
        n_via_alias = len(df[df['matched_via_alias'].str.strip() != ''])
        print(f'\nSummary:')
        print(f'  Total mappings: {len(df)}')
        print(f'  Matched: {n_matched}')
        print(f'  Matched via alias: {n_via_alias}')
        print(f'  Unmatched: {n_unmatched}')
        
        # Save to file
        config.save_new_file(
            df, 
            filepath_dict['make_mapping'], 
            append_to_old=True, 
            timeout=config.CONCURRENT_WRITE_TIMEOUT_LONG, 
            unique_cols=['make_source'], 
            tiebreak_cols=['make_target']
        )
        
        return len(df)
    
    return 0


if __name__ == "__main__":
    # Get source file names (can be multiple or all files)
    source_rumps = config.get_source_rump()
    
    if not source_rumps:
        print('No source files to process.')
        exit()
    
    if isinstance(source_rumps, str):
        source_rumps = [source_rumps]
    
    print(f'\nProcessing {len(source_rumps)} source file(s)...')
    
    total_new_mappings = 0
    for source_rump in source_rumps:
        filepath_dict = config.get_filepaths(source_rump)
        
        # Check if source file exists
        if not os.path.exists(filepath_dict['source']):
            print(f'\nWarning: Source file not found: {filepath_dict["source"]}')
            continue
        
        n_mappings = process_one_source_file(source_rump, filepath_dict)
        total_new_mappings += n_mappings
    
    print(f'\n{"="*60}')
    print(f'COMPLETE: Processed {len(source_rumps)} file(s), created {total_new_mappings} new mappings')
    print(f'{"="*60}')
