import json
import openai
from openai import OpenAI
import config

def map_items_to_targets(source_items, target_items, api_key=None):
    """
    Simple function to map items from source list to target list using LLM.
    
    Args:
        source_items: List of items to map
        target_items: List of target items to map to
        api_key: OpenAI API key (if None, reads from config)
    
    Returns:
        List of dictionaries with mapping results
    """
    # Get API key
    if api_key is None:
        with open(config.OPENAI_KEY_FILEPATH, 'r') as file:
            api_key = file.read().strip()
    
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Create simple prompt
    system_prompt = f"""You are a mapping assistant. Map each source item to the best matching target item.

Available target items:
{chr(10).join([f"- {item}" for item in target_items])}

Return a JSON list where each item has:
- "source": the original source item
- "target": the matched target item (must be exactly from the target list above)
- "confidence": confidence score 0.0 to 1.0

Example format:
[
    {{"source": "item1", "target": "target_item1", "confidence": 0.9}},
    {{"source": "item2", "target": "target_item2", "confidence": 0.8}}
]"""

    user_prompt = f"Map these source items: {source_items}"
    
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
        result = json.loads(content)
        
        # Basic validation - ensure it's a list
        if not isinstance(result, list):
            raise ValueError("LLM response is not a list")
        
        # Validate each mapping
        validated_mappings = []
        for mapping in result:
            # Ensure required fields exist
            if not all(key in mapping for key in ["source", "target", "confidence"]):
                continue
            
            # Validate target is in target list
            if mapping["target"] not in target_items:
                continue
            
            # Validate confidence is numeric
            try:
                confidence = float(mapping["confidence"])
                if not (0.0 <= confidence <= 1.0):
                    continue
            except (ValueError, TypeError):
                continue
            
            validated_mappings.append(mapping)
        
        return validated_mappings
        
    except Exception as e:
        print(f"Error in LLM call: {e}")
        return []

# Example usage
if __name__ == "__main__":
    # Example data
    source_items = ["item1", "item2", "item3"]
    target_items = ["target_a", "target_b", "target_c"]
    
    # Run mapping
    results = map_items_to_targets(source_items, target_items)
    print("Mapping results:")
    for result in results:
        print(f"  {result['source']} -> {result['target']} (confidence: {result['confidence']})")
