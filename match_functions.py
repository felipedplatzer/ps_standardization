import pandas as pd
import re
import config
from llm_make_mapper import llm_make_mapper

ORIGINAL_NAME_FILEPATH = "C:\\Users\\FelipePlatzer\\Documents\\Manifold self-pay\\PartsSource\\Development\\Uptime data ingestion\\summa\\bronze_summa_assets.csv"
ORIGINAL_NAME_COLUMN = 'make'
STANDARDIZED_NAME_FILEPATH = "C:\\Users\\FelipePlatzer\\Documents\\Manifold self-pay\\PartsSource\\Development\\Uptime data ingestion\\global\\Polaris MEL export 2025.04.11 _ temp.csv"
STANDARDIZED_NAME_COLUMN = 'Manufacturer'
STARTING_N = 5
MIN_N = 2 #if the first min_n words match, the match is confirmed. if not, it goes to an LLM


def get_chunk(str, n):
    x = str.strip().replace('.',' ').replace('-',' ').replace('/',' ').replace(',',' ')
    x = re.sub(r'\s{2,}', ' ', x)
    x = x.split(' ')[0:n]
    x= ' '.join(x).lower().strip()
    return x



def get_match(original_name, standardized_names):
    # Try exact match first
    for y in standardized_names:
        # Apply make override if needed
        original_name = config.make_override(str(original_name).strip())
        if str(original_name).strip().lower() == str(y).strip().lower():
            #print(f'Exact, {original_name}, {y}')
            return str(y), 'exact'
        else:
            y_proc = y.lower().strip().replace('.',' ').replace('-',' ').replace('/',' ').replace(',',' ')
            y_proc = re.sub(r'\s{2,}', ' ', y_proc).strip()
            o_proc = original_name.lower().strip().replace('.',' ').replace('-',' ').replace('/',' ').replace(',',' ')
            o_proc = re.sub(r'\s{2,}', ' ', o_proc).strip()
            if y_proc == o_proc:
                #print(f'Skip special chars, {original_name}, {y}')
                return str(y), 'skip_special_chars'
    # Try matching first n words
    n = STARTING_N
    while n >= MIN_N:
        original_chunk = get_chunk(original_name, n)
        for i_std, std_name in enumerate(standardized_names):
            std_chunk = get_chunk(std_name, n)
            if original_chunk == std_chunk and len(original_chunk) >= 4: #eliminate very short words like 'A, or AB'
                #Check if there's another match - if not, it's an unambiguous match. Only check against the next element cause list is sorted
                if i_std == len(standardized_names) - 1:
                    #print(f'first_{str(n)}_words, {original_name}, {std_name}')
                    return std_name, f'first_{str(n)}_words'
                else:
                    next_option = get_chunk(standardized_names[i_std + 1], n)
                    if original_chunk == next_option:
                        pass
                    else:
                        #print(f'first_{str(n)}_words, {original_name}, {std_name}')
                        return std_name, f'first_{str(n)}_words'
        # Take shorter n-grams
        n = n - 1 
    #Default 
    return '', 'no_match'





def get_all_matches(standardized_names, original_names):
    # deterministic matches
    matches = []
    
    for i, original_name in enumerate(original_names):
        if i % 100 == 0:
            x = len([x for x in matches if x['match_type'] != 'no_match'])
            print(f"processed {str(i)} out of {str(len(original_names))} raw names. Found {str(x)} deterministic matches")
        #print(f"processing {str(i)} out of {str(len(original_names))} original names")
        standard_name, match_type = get_match(original_name, standardized_names)
        matches.append({'make_source': original_name, 'make_target': standard_name, 'match_type': match_type}) # must be manually confirmed. Defualt to False
    # LLM matches
    matched_names_dl = [x for x in matches if x['match_type'] != 'no_match']
    unmmatched_names_dl = [x for x in matches if x['match_type'] == 'no_match']
    unmatched_names = [x['make_source'] for x in unmmatched_names_dl]   
    print("Found " + str(len(matched_names_dl)) + " matches out of " + str(len(original_names)) + " original names with deterministic matching")
    print(f"{str(len(unmmatched_names_dl))} unmatched names remaining")
    if len(unmatched_names) > 0:
        matched_names_second_pass_dl = []
        standardized_names_dict = config.tile_by_first_letter(standardized_names)
        unmatched_names_dict = config.tile_by_first_letter(unmatched_names)
        for letter, unmatched_names_subset in unmatched_names_dict.items():
            print(f"Getting LLM matches for makes starting with {letter}")
            standardized_names_subset = standardized_names_dict[letter]
            sub_dl = llm_make_mapper(standardized_names_subset, unmatched_names_subset) #removes unmatched or errors
            matched_names_second_pass_dl.extend(sub_dl)
    else:
        matched_names_second_pass_dl = []
    matched_names_dl = matched_names_dl + matched_names_second_pass_dl
    print(f"Found {str(len(matched_names_second_pass_dl))} matches out of {str(len(unmatched_names))} unmatched names with LLM matching")
    unmatched_names_second_pass_dl = [x for x in unmmatched_names_dl if x['make_source'] not in matched_names_second_pass_dl]
    all_results = matched_names_dl + matched_names_second_pass_dl + unmatched_names_second_pass_dl
    return all_results



"""
def get_input_lists():
    original_names_source = config.read_clean_csv(ORIGINAL_NAME_FILEPATH)[ORIGINAL_NAME_COLUMN].fillna('')
    original_names = list(original_names_source.unique())
    standardized_names_source = config.read_clean_csv(STANDARDIZED_NAME_FILEPATH)[STANDARDIZED_NAME_COLUMN].fillna('')
    standardized_names = list(standardized_names_source.unique())
    return sorted(original_names), sorted(standardized_names)
"""