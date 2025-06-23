import os
import json
import pandas as pd
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator
import openai
from openai import OpenAI
import logging
from datetime import datetime
import time

BATCH_SIZE = 50

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMapping(BaseModel):
    """Pydantic model for individual model mapping"""
    model_name_source: str = Field(..., description="The raw model name from source data")
    model_name_target: str = Field(..., description="The standardized model name from target list")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the mapping (0-1)")
    reasoning: str = Field(..., description="Brief explanation of why this mapping was chosen")

class ModelMappingBatch(BaseModel):
    """Pydantic model for batch of model mappings"""
    mappings: List[ModelMapping] = Field(..., description="List of model mappings")
    total_processed: int = Field(..., description="Total number of models processed in this batch")
    successful_mappings: int = Field(..., description="Number of successful mappings")

class LLMModelMapper:
    """Class to handle LLM-based model mapping with validation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", max_retries: int = 3):
        """
        Initialize the LLM Model Mapper
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (default: gpt-4o)
            max_retries: Maximum number of retries for API calls
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.model_name_target_list = []
        
    def set_model_name_target_list(self, model_name_target_list: List[str]):
        """Set the list of standardized models for validation"""
        self.model_name_target_list = [str(model).strip() for model in model_name_target_list if str(model).strip()]
        #logger.info(f"Set {len(self.model_name_target_list)} standardized models for validation")
        
    def validate_model_name_target(self, model: str) -> bool:
        """Validate that a model exists in the standardized list"""
        return model.strip() in self.model_name_target_list
    
    def create_system_prompt(self) -> str:
        """Create the system prompt for the LLM"""
        return f"""You are an expert equipment model mapper. Your task is to map raw equipment model names to standardized model names from a predefined list.

Available standardized models:
{chr(10).join([f"- {model}" for model in self.model_name_target_list])}

Instructions:
1. Analyze each raw model name carefully
2. Find the best match from the standardized list
3. Consider variations in naming conventions, abbreviations, and formatting
4. If no good match exists, use "NO_MATCH" as the standardized model
5. Provide a confidence score (0.0 to 1.0) for your mapping
6. Explain your reasoning briefly

IMPORTANT: You must ONLY use model names from the standardized list above. If you cannot find a good match, use "NO_MATCH".

Output format (JSON):
{{
    "mappings": [
        {{
            "model_name_source": "original model name",
            "model_name_target": "matched standardized model or NO_MATCH",
            "confidence": 0.95,
            "reasoning": "Brief explanation"
        }}
    ]
}}"""

    def create_user_prompt(self, model_name_source_list: List[str]) -> str:
        """Create the user prompt with raw models to map"""
        models_text = chr(10).join([f"- {model}" for model in model_name_source_list])
        return f"""Please map the following raw equipment model names to standardized models:

{models_text}

Return your response as a valid JSON object following the specified format."""

    def call_openai_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Make API call to OpenAI with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistent results
                    max_tokens=4000,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)
            except Exception as e:
                logger.warning(f"API call error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
                
    def map_models(self, model_name_source_list: List[str], batch_size: int = BATCH_SIZE) -> ModelMappingBatch:
        """
        Map raw models to standardized models using LLM
        
        Args:
            model_name_source_list: List of raw model names to map
            batch_size: Number of models to process in each batch
            
        Returns:
            ModelMappingBatch with all mappings
        """
        if not self.model_name_target_list:
            raise ValueError("Standardized models list not set. Call set_model_name_target_list() first.")
            
        all_mappings = []
        total_processed = 0
        
        # Process in batches
        for i in range(0, len(model_name_source_list), batch_size):
            batch = model_name_source_list[i:i + batch_size]
            #logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} models")
            
            try:
                # Create prompts
                system_prompt = self.create_system_prompt()
                user_prompt = self.create_user_prompt(batch)
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # Call OpenAI API
                response = self.call_openai_api(messages)
                
                # Validate and process response
                batch_mappings = []
                for mapping_data in response.get("mappings", []):
                    try:
                        # Validate that standardized model is in our list
                        model_name_target = mapping_data.get("model_name_target", "").strip()
                        if model_name_target != "NO_MATCH" and not self.validate_model_name_target(model_name_target):
                            logger.warning(f"Invalid standardized model '{model_name_target}' for raw model '{mapping_data.get('model_name_source')}'. Setting to NO_MATCH.")
                            mapping_data["model_name_target"] = "NO_MATCH"
                            mapping_data["confidence"] = 0.0
                            mapping_data["reasoning"] = "Invalid standardized model - not in allowed list"
                        
                        # Create Pydantic model
                        mapping = ModelMapping(**mapping_data)
                        batch_mappings.append(mapping)
                        
                    except Exception as e:
                        logger.error(f"Error processing mapping for {mapping_data.get('model_name_source', 'unknown')}: {e}")
                        # Create a fallback mapping
                        fallback_mapping = ModelMapping(
                            model_name_source=mapping_data.get("model_name_source", "unknown"),
                            model_name_target="NO_MATCH",
                            confidence=0.0,
                            reasoning=f"Error in processing: {str(e)}",
                        )
                        batch_mappings.append(fallback_mapping)
                
                all_mappings.extend(batch_mappings)
                total_processed += len(batch)
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Create fallback mappings for this batch
                for model_name_source in batch:
                    fallback_mapping = ModelMapping(
                        model_name_source=model_name_source,
                        model_name_target="NO_MATCH",
                        confidence=0.0,
                        reasoning=f"Batch processing error: {str(e)}",
                    )
                    all_mappings.append(fallback_mapping)
                total_processed += len(batch)
        
        # Create final batch result
        successful_mappings = len([m for m in all_mappings if m.model_name_target != "NO_MATCH"])
        
        return ModelMappingBatch(
            mappings=all_mappings,
            total_processed=total_processed,
            successful_mappings=successful_mappings
        )
    
    def mappings_to_dict_list(self, mapping_batch: ModelMappingBatch) -> List[Dict[str, Any]]:
        """Convert mappings to list of dictionaries"""
        data = []
        for mapping in mapping_batch.mappings:
            data.append({
                'model_name_source': mapping.model_name_source,
                'model_name_target': mapping.model_name_target,
                'confidence': mapping.confidence,
                'reasoning': mapping.reasoning,
            })
        
        return data
    
    def get_mapping_summary(self, mapping_batch: ModelMappingBatch) -> Dict[str, Any]:
        """Get a summary of the mapping results"""
        total = len(mapping_batch.mappings)
        successful = mapping_batch.successful_mappings
        failed = total - successful
        
        confidence_ranges = {
            'high': len([m for m in mapping_batch.mappings if m.confidence >= 0.8]),
            'medium': len([m for m in mapping_batch.mappings if 0.5 <= m.confidence < 0.8]),
            'low': len([m for m in mapping_batch.mappings if m.confidence < 0.5])
        }
        
        return {
            'total_models': total,
            'successful_mappings': successful,
            'failed_mappings': failed,
            'success_rate': successful / total if total > 0 else 0,
            'confidence_ranges': confidence_ranges,
        }


def main(model_name_target_list: List[str], model_name_source_list: List[str]) -> List[Dict[str, Any]]:
    """
    Main function to map raw models to standardized models using LLM
    
    Args:
        model_name_target_list: List of standardized model names to match against
        model_name_source_list: List of raw model names to map
    
    Returns:
        List of dictionaries with mapping results
    """
    
    # Initialize mapper
    mapper = LLMModelMapper(api_key='sk-2PLrQ8sUdX77fM_rDH83nGuW4az2O94jpeLH_aRDN8T3BlbkFJNZZYtTNu3HVSaEIW04_3YFyOkumlmyP5feAbS0VSgA')
    
    # Set standardized models
    mapper.set_model_name_target_list(model_name_target_list)
    
    # Perform mapping
    print("Starting model mapping...")
    mapping_batch = mapper.map_models(model_name_source_list, batch_size=BATCH_SIZE)
    
    # Print summary
    summary = mapper.get_mapping_summary(mapping_batch)
    #print("\nMapping Summary:")
    #print(f"Total models: {summary['total_models']}")
    #print(f"Successful mappings: {summary['successful_mappings']}")
    #print(f"Success rate: {summary['success_rate']:.2%}")
    #print(f"Confidence ranges: {summary['confidence_ranges']}")
    
    # Return list of dictionaries
    return mapper.mappings_to_dict_list(mapping_batch)
