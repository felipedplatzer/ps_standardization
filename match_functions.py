import pandas as pd
import re

ORIGINAL_NAME_FILEPATH = "C:\\Users\\FelipePlatzer\\Documents\\Manifold self-pay\\PartsSource\\Development\\Uptime data ingestion\\summa\\bronze_summa_assets.csv"
ORIGINAL_NAME_COLUMN = 'make'
STANDARDIZED_NAME_FILEPATH = "C:\\Users\\FelipePlatzer\\Documents\\Manifold self-pay\\PartsSource\\Development\\Uptime data ingestion\\global\\Polaris MEL export 2025.04.11 _ temp.csv"
STANDARDIZED_NAME_COLUMN = 'Manufacturer'
STARTING_N = 5

def get_input_lists():
    original_names_source = pd.read_csv(ORIGINAL_NAME_FILEPATH, encoding='utf-8')[ORIGINAL_NAME_COLUMN].fillna('')
    original_names = list(original_names_source.unique())
    standardized_names_source = pd.read_csv(STANDARDIZED_NAME_FILEPATH, encoding='utf-8')[STANDARDIZED_NAME_COLUMN].fillna('')
    standardized_names = list(standardized_names_source.unique())
    return sorted(original_names), sorted(standardized_names)


def get_chunk(str, n):
    x = str.strip().replace('.',' ').replace('-',' ').replace('/',' ').replace(',',' ')
    x = re.sub(r'\s{2,}', ' ', x)
    x = x.split(' ')[0:n]
    x= ' '.join(x).lower().strip()
    return x



def get_match(original_name, standardized_names):
    # Try exact match first
    for y in standardized_names:
        if str(original_name).strip().lower() == str(y).strip().lower():
            #print(f'Exact, {original_name}, {y}')
            return str(y), 'exact', True
        else:
            y_proc = y.lower().strip().replace('.',' ').replace('-',' ').replace('/',' ').replace(',',' ')
            y_proc = re.sub(r'\s{2,}', ' ', y_proc).strip()
            o_proc = original_name.lower().strip().replace('.',' ').replace('-',' ').replace('/',' ').replace(',',' ')
            o_proc = re.sub(r'\s{2,}', ' ', o_proc).strip()
            if y_proc == o_proc:
                #print(f'Skip special chars, {original_name}, {y}')
                return str(y), 'skip_special_chars', False
    # Try matching first n words
    n = STARTING_N
    while n >= 1:
        original_chunk = get_chunk(original_name, n)
        for i_std, std_name in enumerate(standardized_names):
            std_chunk = get_chunk(std_name, n)
            if original_chunk == std_chunk and len(original_chunk) >= 4: #eliminate very short words like 'A, or AB'
                #Check if there's another match - if not, it's an unambiguous match. Only check against the next element cause list is sorted
                if i_std == len(standardized_names) - 1:
                    #print(f'first_{str(n)}_words, {original_name}, {std_name}')
                    return std_name, f'first_{str(n)}_words', False
                else:
                    next_option = get_chunk(standardized_names[i_std + 1], n)
                    if original_chunk == next_option:
                        pass
                    else:
                        #print(f'first_{str(n)}_words, {original_name}, {std_name}')
                        return std_name, f'first_{str(n)}_words', False
        # Take shorter n-grams
        n = n - 1 
    #Default 
    return '', 'no_match', False


def get_all_matches(standardized_names, original_names):
    matches = []
    for i, original_name in enumerate(original_names):
        if i % 100 == 0:
            x = len([x for x in matches if x['match_type'] != 'no_match'])
            print(f"processed {str(i)} out of {str(len(original_names))} raw names. Found {str(x)} matches")
        #print(f"processing {str(i)} out of {str(len(original_names))} original names")
        standard_name, match_type, confirmed = get_match(original_name, standardized_names)
        matches.append({'make_source': original_name, 'make_target': standard_name, 'match_type': match_type, 'confirmed': confirmed}) # must be manually confirmed. Defualt to False
    return matches
