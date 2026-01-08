"""
Hierarchical Modality Mapper

Maps devices to modalities in a hierarchical fashion:
1. First map to L1 (most general - like continent)
2. Then map to L2 (filtered by known L1 - like country)
3. Then map to L3 (filtered by known L1 and L2 - like city)
"""

import json
import pandas as pd
from typing import List, Dict, Any, Optional
from openai import OpenAI
import logging
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = 500


def build_taxonomy_lookup(mel_df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    """
    Build a taxonomy lookup structure from MEL data.
    
    Returns:
        Dict with structure:
        {
            'l1_list': ['L1_A', 'L1_B', ...],
            'l1_to_l2': {'L1_A': ['L2_A1', 'L2_A2'], 'L1_B': ['L2_B1'], ...},
            'l1_l2_to_l3': {('L1_A', 'L2_A1'): ['L3_A1a', 'L3_A1b'], ...},
            'l3_to_l1_l2': {'L3_A1a': ('L1_A', 'L2_A1'), ...}
        }
    """
    taxonomy = {
        'l1_list': [],
        'l1_to_l2': {},
        'l1_l2_to_l3': {},
        'l3_to_l1_l2': {}
    }
    
    # Get unique combinations
    for _, row in mel_df.iterrows():
        l1 = str(row.get('l1_modality_target', '')).strip()
        l2 = str(row.get('l2_modality_target', '')).strip()
        l3 = str(row.get('l3_modality_target', '')).strip()
        
        if l1 and l1 not in taxonomy['l1_list']:
            taxonomy['l1_list'].append(l1)
        
        if l1 and l2:
            if l1 not in taxonomy['l1_to_l2']:
                taxonomy['l1_to_l2'][l1] = []
            if l2 not in taxonomy['l1_to_l2'][l1]:
                taxonomy['l1_to_l2'][l1].append(l2)
        
        if l1 and l2 and l3:
            key = (l1, l2)
            if key not in taxonomy['l1_l2_to_l3']:
                taxonomy['l1_l2_to_l3'][key] = []
            if l3 not in taxonomy['l1_l2_to_l3'][key]:
                taxonomy['l1_l2_to_l3'][key].append(l3)
            
            taxonomy['l3_to_l1_l2'][l3] = (l1, l2)
    
    # Sort lists for consistency
    taxonomy['l1_list'] = sorted(taxonomy['l1_list'])
    for l1 in taxonomy['l1_to_l2']:
        taxonomy['l1_to_l2'][l1] = sorted(taxonomy['l1_to_l2'][l1])
    for key in taxonomy['l1_l2_to_l3']:
        taxonomy['l1_l2_to_l3'][key] = sorted(taxonomy['l1_l2_to_l3'][key])
    
    logger.info(f"Built taxonomy with {len(taxonomy['l1_list'])} L1 categories, "
                f"{sum(len(v) for v in taxonomy['l1_to_l2'].values())} L2 categories, "
                f"{sum(len(v) for v in taxonomy['l1_l2_to_l3'].values())} L3 categories")
    
    return taxonomy


def create_modality_prompt(devices: List[Dict], modality_list: List[str], level: str, 
                           known_l1: Optional[str] = None, known_l2: Optional[str] = None) -> tuple:
    """
    Create system and user prompts for modality classification.
    
    Args:
        devices: List of device dicts with make_target and model_name_source
        modality_list: List of valid modalities for this level
        level: 'L1', 'L2', or 'L3'
        known_l1: Known L1 modality (for L2 and L3 mapping)
        known_l2: Known L2 modality (for L3 mapping)
    """
    context = ""
    if level == "L2" and known_l1:
        context = f"\nThe L1 category is already known to be: {known_l1}\nYou must select from L2 categories that belong to this L1."
    elif level == "L3" and known_l1 and known_l2:
        context = f"\nThe L1 category is: {known_l1}\nThe L2 category is: {known_l2}\nYou must select from L3 categories that belong to this L1/L2 combination."
    
    system_prompt = f"""You are an expert medical device classifier. Your task is to classify medical devices into {level} modality categories based on their manufacturer (make) and model name.
{context}
Available {level} modalities:
{chr(10).join([f"- {modality}" for modality in modality_list])}

Instructions:
1. Analyze each device's make and model name carefully
2. Consider the manufacturer's typical product lines and the model naming conventions
3. Classify the device into the most appropriate {level} modality from the list above
4. Provide a confidence score (0.0 to 1.0) for your classification

IMPORTANT: You must ONLY use modalities from the list above. If you cannot determine a match, use "NO_MATCH".

Output format (JSON):
{{
    "classifications": [
        {{
            "make": "manufacturer name",
            "model_name": "model name",
            "modality": "classified {level} modality or NO_MATCH",
            "confidence": 0.95
        }}
    ]
}}"""

    devices_text = chr(10).join([f"- Make: {d.get('make_target', '')}, Model: {d.get('model_name_source', '')}" for d in devices])
    user_prompt = f"""Please classify the following medical devices into {level} modality categories:

{devices_text}

Return your response as a valid JSON object following the specified format."""

    return system_prompt, user_prompt


def call_llm_for_modality(devices: List[Dict], modality_list: List[str], level: str,
                          known_l1: Optional[str] = None, known_l2: Optional[str] = None,
                          api_key: Optional[str] = None) -> List[Dict]:
    """
    Call LLM to classify devices into modalities.
    
    Returns:
        List of dicts with 'make', 'model_name', 'modality', 'confidence'
    """
    if not devices or not modality_list:
        return []
    
    if api_key is None:
        with open(config.OPENAI_KEY_FILEPATH, 'r') as file:
            api_key = file.read().strip()
    
    client = OpenAI(api_key=api_key)
    system_prompt, user_prompt = create_modality_prompt(devices, modality_list, level, known_l1, known_l2)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=config.MODALITY_MAPPER_TEMPERATURE,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        classifications = result.get("classifications", [])
        
        # Validate modalities
        validated = []
        for c in classifications:
            modality = c.get("modality", "").strip()
            if modality == "NO_MATCH" or modality not in modality_list:
                c["modality"] = ""
                c["confidence"] = 0.0
            validated.append(c)
        
        return validated
        
    except Exception as e:
        logger.error(f"Error in LLM call for {level}: {e}")
        # Return empty modalities for all devices
        return [{"make": d.get("make_target", ""), "model_name": d.get("model_name_source", ""), 
                 "modality": "", "confidence": 0.0} for d in devices]


def hierarchical_modality_mapping(devices_df: pd.DataFrame, taxonomy: Dict) -> pd.DataFrame:
    """
    Perform hierarchical modality mapping: L1 -> L2 -> L3.
    
    For each unmatched device:
    1. First classify into L1 (from all L1 options)
    2. Then classify into L2 (from L2s that belong to the assigned L1)
    3. Then classify into L3 (from L3s that belong to the assigned L1+L2)
    
    Args:
        devices_df: DataFrame with devices to classify (must have 'make_target', 'model_name_source')
        taxonomy: Taxonomy lookup from build_taxonomy_lookup()
    
    Returns:
        DataFrame with added columns: l1_modality_target, l2_modality_target, l3_modality_target
    """
    if len(devices_df) == 0:
        return devices_df
    
    # Initialize modality columns if not present
    for col in ['l1_modality_target', 'l2_modality_target', 'l3_modality_target',
                'l1_modality_confidence', 'l2_modality_confidence', 'l3_modality_confidence',
                'modality_match_type']:
        if col not in devices_df.columns:
            devices_df[col] = ''
    
    devices_dl = devices_df.to_dict(orient='records')
    
    # STEP 1: Map to L1
    logger.info(f"Step 1: Mapping {len(devices_dl)} devices to L1 categories...")
    l1_list = taxonomy['l1_list']
    
    # Process in batches
    for i in range(0, len(devices_dl), BATCH_SIZE):
        batch = devices_dl[i:i + BATCH_SIZE]
        logger.info(f"  Processing L1 batch {i//BATCH_SIZE + 1}: {len(batch)} devices")
        
        l1_results = call_llm_for_modality(batch, l1_list, "L1")
        
        # Match results back to devices
        for j, result in enumerate(l1_results):
            if i + j < len(devices_dl):
                devices_dl[i + j]['l1_modality_target'] = result.get('modality', '')
                devices_dl[i + j]['l1_modality_confidence'] = result.get('confidence', 0.0)
    
    # STEP 2: Map to L2 (grouped by L1)
    logger.info("Step 2: Mapping devices to L2 categories (grouped by L1)...")
    
    # Group devices by L1
    l1_groups = {}
    for d in devices_dl:
        l1 = d.get('l1_modality_target', '')
        if l1 and l1 in taxonomy['l1_to_l2']:
            if l1 not in l1_groups:
                l1_groups[l1] = []
            l1_groups[l1].append(d)
    
    # Process each L1 group
    for l1, group_devices in l1_groups.items():
        l2_list = taxonomy['l1_to_l2'].get(l1, [])
        if not l2_list:
            continue
        
        logger.info(f"  Mapping {len(group_devices)} devices for L1='{l1}' to {len(l2_list)} L2 categories")
        
        for i in range(0, len(group_devices), BATCH_SIZE):
            batch = group_devices[i:i + BATCH_SIZE]
            l2_results = call_llm_for_modality(batch, l2_list, "L2", known_l1=l1)
            
            for j, result in enumerate(l2_results):
                if i + j < len(group_devices):
                    group_devices[i + j]['l2_modality_target'] = result.get('modality', '')
                    group_devices[i + j]['l2_modality_confidence'] = result.get('confidence', 0.0)
    
    # STEP 3: Map to L3 (grouped by L1+L2)
    logger.info("Step 3: Mapping devices to L3 categories (grouped by L1+L2)...")
    
    # Group devices by L1+L2
    l1_l2_groups = {}
    for d in devices_dl:
        l1 = d.get('l1_modality_target', '')
        l2 = d.get('l2_modality_target', '')
        if l1 and l2 and (l1, l2) in taxonomy['l1_l2_to_l3']:
            key = (l1, l2)
            if key not in l1_l2_groups:
                l1_l2_groups[key] = []
            l1_l2_groups[key].append(d)
    
    # Process each L1+L2 group
    for (l1, l2), group_devices in l1_l2_groups.items():
        l3_list = taxonomy['l1_l2_to_l3'].get((l1, l2), [])
        if not l3_list:
            continue
        
        logger.info(f"  Mapping {len(group_devices)} devices for L1='{l1}', L2='{l2}' to {len(l3_list)} L3 categories")
        
        for i in range(0, len(group_devices), BATCH_SIZE):
            batch = group_devices[i:i + BATCH_SIZE]
            l3_results = call_llm_for_modality(batch, l3_list, "L3", known_l1=l1, known_l2=l2)
            
            for j, result in enumerate(l3_results):
                if i + j < len(group_devices):
                    group_devices[i + j]['l3_modality_target'] = result.get('modality', '')
                    group_devices[i + j]['l3_modality_confidence'] = result.get('confidence', 0.0)
    
    # Set modality_match_type for all processed devices
    for d in devices_dl:
        if d.get('l1_modality_target') or d.get('l2_modality_target') or d.get('l3_modality_target'):
            d['modality_match_type'] = 'llm_hierarchical'
    
    # Convert back to DataFrame
    result_df = pd.DataFrame(devices_dl)
    
    # Ensure all modality columns are strings
    for col in ['l1_modality_target', 'l2_modality_target', 'l3_modality_target', 'modality_match_type']:
        if col in result_df.columns:
            result_df[col] = result_df[col].fillna('').astype(str)
    
    return result_df


def map_unmatched_devices(make_df: pd.DataFrame, taxonomy: Dict) -> pd.DataFrame:
    """
    Map modalities for devices that didn't get a MEL match.
    For unmatched devices where LLM mapping also fails, fills l3_modality_target with l3_modality_source.
    
    Args:
        make_df: DataFrame with all devices for a make
        taxonomy: Taxonomy lookup
    
    Returns:
        DataFrame with modalities filled in for unmatched devices
    """
    # Split into matched and unmatched
    matched_mask = make_df['model_match_type'].str.lower() != 'no_match'
    matched_df = make_df[matched_mask].copy()
    unmatched_df = make_df[~matched_mask].copy()
    
    if len(unmatched_df) == 0:
        # All matched - modalities come from MEL backfill
        matched_df['modality_match_type'] = 'from_mel_based_on_model'
        return matched_df
    
    # Set match type for matched devices
    matched_df['modality_match_type'] = 'from_mel_based_on_model'
    
    # Perform hierarchical mapping for unmatched devices
    logger.info(f"Performing hierarchical modality mapping for {len(unmatched_df)} unmatched devices")
    unmatched_df = hierarchical_modality_mapping(unmatched_df, taxonomy)
    
    # For devices still without l3_modality_target after LLM mapping, use source value
    if 'l3_modality_target' in unmatched_df.columns and 'l3_modality_source' in unmatched_df.columns:
        empty_l3_mask = unmatched_df['l3_modality_target'].fillna('').str.strip() == ''
        unmatched_df.loc[empty_l3_mask, 'l3_modality_target'] = unmatched_df.loc[empty_l3_mask, 'l3_modality_source']
        # Update match type for those filled with source
        unmatched_df.loc[empty_l3_mask, 'modality_match_type'] = 'use_source_value'
    
    if 'l3_modality_target_normalized' in unmatched_df.columns and 'l3_modality_source_normalized' in unmatched_df.columns:
        empty_l3_norm_mask = unmatched_df['l3_modality_target_normalized'].fillna('').str.strip() == ''
        unmatched_df.loc[empty_l3_norm_mask, 'l3_modality_target_normalized'] = unmatched_df.loc[empty_l3_norm_mask, 'l3_modality_source_normalized']
    
    # Combine matched and unmatched
    result_df = pd.concat([matched_df, unmatched_df], ignore_index=True)
    
    return result_df
