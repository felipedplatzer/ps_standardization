import pandas as pd
import json
from openai import OpenAI
import config

def get_unique_source_modalities(df):
    df['modality_source'] = df['modality_source'].fillna('')
    source_modality_list = list(df['modality_source'].unique())
    source_modality_list = [modality for modality in source_modality_list if modality.strip() != '']
    return source_modality_list

def create_system_prompt(target_items):
    return f"""You are an expert equipment model mapper. Your task is to map raw equipment model names to standardized model names from a predefined list.

    Available standardized models:
    {chr(10).join([f"- {model}" for model in target_items])}

    Instructions:
    1. Find the best match from the standardized list
    2. Consider variations in naming conventions, abbreviations, and formatting
    3. Provide a confidence score (0.0 to 1.0) for your mapping
    4. If no match, return "NO_MATCH"

    IMPORTANT: You must ONLY use model names from the standardized list above.

    Output format (JSON):
    {{
        "mappings": [
            {{
                "category_raw": "original model name",
                "category_standardized": "matched standardized model or NO_MATCH",
                "confidence": 0.95,
            }}
        ]
    }}"""


def create_user_prompt(source_items):
    models_text = chr(10).join([f"- {model}" for model in source_items])
    return f"""Please map the following raw equipment model names to standardized models:

    {models_text}

    Return your response as a valid JSON object following the specified format."""


def run_llm(source_items, target_items, api_key=None):
    # Get API key
    if api_key is None:
        with open(config.OPENAI_KEY_FILEPATH, 'r') as file:
            api_key = file.read().strip()
    if not api_key:
        raise ValueError("OpenAI API key is required")
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Create simple prompt
    system_prompt = create_system_prompt(target_items)
    user_prompt = create_user_prompt(source_items)

    # Make LLM call
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        content = response.choices[0].message.content
        mappings = json.loads(content)["mappings"]
        
        # Basic validation - ensure it's a list
        if not isinstance(mappings, list):
            raise ValueError("LLM response is not a list")
        
        # Validate each mapping
        for mapping in mappings:
            # Validate target is in target list
            if mapping["category_standardized"] not in target_items:
                mapping["category_standardized"] = ""
                
                    
        return mappings
        
    except Exception as e:
        print(f"Error in LLM call: {e}")
        return []







def second_stage_modality_mapping(input_df, target_modality_list):
    #Get modalities (source)
    source_modality_list = get_unique_source_modalities(input_df)
    #Map modalities (source) to target modalities (MEL)
    modality_dict_list = run_llm(source_modality_list, target_modality_list)
    #Add back removed columns
    return modality_dict_list

"""
target_df = pd.read_csv('./karl_storz_test_2025-08-22/mel.csv')
target_modality_list = list(target_df['New Lvl 2 Category'].unique().astype(str))
df = pd.read_csv("./karl_storz_test_2025-08-22/source.csv")
df = second_stage_modality_mapping(df, target_modality_list)
df.to_csv("./test_2025-08-22.csv", index=False)"""



"""
def add_back_removed_columns(input_df, modality_dict_list):
    input_df = input_df.drop_columns(['modality_target']) #avoid duplicate column . ps_modality from df is blank (since df contains only unmatched modalities
    modality_df = pd.DataFrame(modality_dict_list)
    output_df = pd.merge(input_df, modality_df, on='modality_source', how='left').fillna('')
    return output_df"""