import openai
import json
import config

def llm_make_mapper(standardized_names, original_names):
    """
    Standardize company names using LLM against a list of standardized names.
    
    Args:
        standardized_names: List of valid standardized company names
        original_names: List of raw company names to standardize
    
    Returns:
        List of dicts with 'company_raw' and 'company_standardized' keys
    """
    
    # Load API key
    with open(config.OPENAI_KEY_FILEPATH, 'r') as file:
        api_key = file.read().strip()
    
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Create OpenAI client with new API
    client = openai.OpenAI(api_key=api_key)
    
    results = []

    original_names_text = chr(10).join([f"- {company}" for company in original_names])
    standardized_names_text = chr(10).join([f"- {company}" for company in standardized_names])



    # Create prompt for the LLM
    prompt = f"""
    You are a helpful assistant that standardizes names of medical device companies.  You will be given a list of standardized company names and a list of raw company names.  
    You will need to return a list of raw company names and the corresponding standardized company name that best matches the raw company name.
    
    Standardized company names: {standardized_names_text}

    Raw company names: {original_names_text}
    
    Return ONLY a JSON object with this exact format:
    {{
        "matches": [
            {{
                "company_raw": "raw company name",
                "company_standardized": "standardized company name"
            }}
        ]
    }}
    
    The company_standardized value MUST be one of the standardized names from the list above.
    If no good match exists, use an empty string "".
    """
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=config.TEMPERATURE,
        response_format={"type": "json_object"}
    )
    
    # Extract response content
    content = response.choices[0].message.content.strip()
    # Parse JSON response
    results = json.loads(content)['matches']
    results_processed = []
    for x in results:
        try: 
            if x['company_standardized'] in standardized_names and x['company_standardized'] != '':
                x_processed = {
                    'make_source': x['company_raw'],
                    'make_target': x['company_standardized'],
                    'match_type': 'llm'
                }
                results_processed.append(x_processed)
        except:
            pass
    
    return results_processed


