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
import config


BATCH_SIZE = 1000
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Device(BaseModel):
    """Pydantic model for individual device"""
    make: str = Field(..., description="The manufacturer/make of the device")
    model_name: str = Field(..., description="The model name of the device")
    modality: str = Field(..., description="The raw modality of the device")
    
class DeviceClassification(BaseModel):
    """Pydantic model for individual device classification"""
    make: str = Field(..., description="The manufacturer/make of the device")
    model_name: str = Field(..., description="The model name of the device")
    modality: str = Field(..., description="The classified modality from target list")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the classification (0-1)")

class DeviceClassificationBatch(BaseModel):
    """Pydantic model for batch of device classifications"""
    classifications: List[DeviceClassification] = Field(..., description="List of device classifications")
    total_processed: int = Field(..., description="Total number of devices processed in this batch")
    successful_classifications: int = Field(..., description="Number of successful classifications")

class DeviceClassifier:
    """Class to handle LLM-based device classification with validation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", max_retries: int = 3):
        """
        Initialize the Device Classifier
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (default: gpt-4o)
            max_retries: Maximum number of retries for API calls
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.target_modality_list = []
        
    def set_target_modality_list(self, target_modality_list: List[str]):
        """Set the list of target modalities for validation"""
        self.target_modality_list = [str(modality).strip() for modality in target_modality_list if str(modality).strip()]
        logger.info(f"Set {len(self.target_modality_list)} target modalities for validation")
        
    def validate_modality(self, modality: str) -> bool:
        """Validate that a modality exists in the target list"""
        return modality.strip() in self.target_modality_list
    
    def create_system_prompt(self) -> str:
        """Create the system prompt for the LLM"""
        return f"""You are an expert medical device classifier. Your task is to classify medical devices into pre-defined, standardized modalities based on their manufacturer (make), model name, and "raw" (i.e. unstandardized) modality.

Available modalities:
{chr(10).join([f"- {modality}" for modality in self.target_modality_list])}

Instructions:
1. Analyze each device's make and model name carefully
2. Consider the manufacturer's typical product lines, model name, and the "raw" modality
3. Classify the device into the most appropriate modality from the list above
4. Provide a confidence score (0.0 to 1.0) for your classification

IMPORTANT: You must ONLY use modalities from the list above. The modality may or may not be the same as the "raw" modality.

Output format (JSON):
{{
    "classifications": [
        {{
            "make": "manufacturer name",
            "model_name": "model name",
            "modality": "classified modality or no_match",
            "confidence": 0.95
        }}
    ]
}}"""

    def create_user_prompt(self, devices: List[Device]) -> str:
        """Create the user prompt with devices to classify"""
        devices_text = chr(10).join([f"- Make: {device.make}, Model: {device.model_name}, Raw Modality: {device.modality}" for device in devices])
        return f"""Please classify the following medical devices by modality:

{devices_text}

Return your response as a valid JSON object following the specified format."""

    def call_openai_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Make API call to OpenAI with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=config.MODALITY_MAPPER_TEMPERATURE,  # Low temperature for consistent results
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
                
    def classify_devices(self, devices: List[Device], batch_size: int = BATCH_SIZE) -> DeviceClassificationBatch:
        """
        Classify devices by modality using LLM
        
        Args:
            devices: List of devices to classify
            batch_size: Number of devices to process in each batch
            
        Returns:
            DeviceClassificationBatch with all classifications
        """
        if not self.target_modality_list:
            raise ValueError("Target modality list not set. Call set_target_modality_list() first.")
            
        all_classifications = []
        total_processed = 0
        
        # Process in batches
        for i in range(0, len(devices), batch_size):
            batch = devices[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} devices")
            
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
                batch_classifications = []
                for classification_data in response.get("classifications", []):
                    try:
                        # Validate that modality is in our list
                        modality = classification_data.get("modality", "").strip()
                        if modality != "no_match" and not self.validate_modality(modality):
                            logger.warning(f"Invalid modality '{modality}' for device '{classification_data.get('make_target')} {classification_data.get('model_name_source')}'. Setting to no_match.")
                            classification_data["modality"] = "no_match"
                            classification_data["confidence"] = 0.0
                        
                        # Create Pydantic model
                        classification = DeviceClassification(**classification_data)
                        batch_classifications.append(classification)
                        
                    except Exception as e:
                        logger.error(f"Error processing classification for {classification_data.get('make_target', 'no_match')} {classification_data.get('model_name_source', 'no_match')}: {e}")
                        # Create a fallback classification
                        fallback_classification = DeviceClassification(
                            make=classification_data.get("make", "no_match"),
                            model_name=classification_data.get("model_name", "no_match"),
                            modality="no_match",
                            confidence=0.0,
                        )
                        batch_classifications.append(fallback_classification)
                
                all_classifications.extend(batch_classifications)
                total_processed += len(batch)
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Create fallback classifications for this batch
                for device in batch:
                    fallback_classification = DeviceClassification(
                        make=device.make,
                        model_name=device.model_name,
                        modality="no_match",
                        confidence=0.0,
                    )
                    all_classifications.append(fallback_classification)
                total_processed += len(batch)
        
        # Replace 'no_match' with ''
        for x in all_classifications:
            x.modality = x.modality.replace('no_match', '')

        #  Create final batch result
        successful_classifications = len([c for c in all_classifications if c.modality != "no_match"])
        
        return DeviceClassificationBatch(
            classifications=all_classifications,
            total_processed=total_processed,
            successful_classifications=successful_classifications
        )
    
    def classifications_to_dict_list(self, classification_batch: DeviceClassificationBatch) -> List[Dict[str, Any]]:
        """Convert classifications to list of dictionaries"""
        data = []
        for classification in classification_batch.classifications:
            data.append({
                'make_target': classification.make,
                'model_name_source': classification.model_name,
                'modality_target': classification.modality,
                'modality_confidence': classification.confidence,
            })
        
        return data
    


def main(target_modality_list: List[str], devices: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Main function to classify devices by modality using LLM
    
    Args:
        target_modality_list: List of target modalities to classify into
        devices: List of dictionaries with 'make_target' and 'model_name_source' keys
    
    Returns:
        List of dictionaries with classification results
    """
    
    # Initialize classifier
    # read api key from text file
    with open(config.OPENAI_KEY_FILEPATH, 'r') as file:
        api_key = file.read().strip()
    
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    classifier = DeviceClassifier(api_key=api_key)
    
    # Set target modalities
    classifier.set_target_modality_list(target_modality_list)
    
    # Convert devices to Pydantic models
    device_models = [Device(make=device['make_target'], model_name=device['model_name_source'], modality=device['modality_source']) for device in devices]
    
    # Perform classification
    logger.info("Starting device classification...")
    classification_batch = classifier.classify_devices(device_models, batch_size=BATCH_SIZE)
    
    # Print summary
    logger.info("\nClassification Summary:")
    logger.info(f"Total devices: {classification_batch.total_processed}")
    logger.info(f"Successful classifications: {classification_batch.successful_classifications}")
    
    # Return list of dictionaries
    return classifier.classifications_to_dict_list(classification_batch)


def classify_devices_from_dataframe(devices: List[Dict[str, str]], target_modality_list: List[str], 
                                  make_column: str = 'make_target', model_column: str = 'model_name_source', modality_column: str = 'modality_source') -> pd.DataFrame:
    """
    Convenience function to classify devices from a pandas DataFrame
    
    Args:
        df: DataFrame containing device data
        target_modality_list: List of target modalities to classify into
        make_column: Name of the column containing manufacturer/make
        model_column: Name of the column containing model names
        modality_column: Name of the column containing the modality
    Returns:
        DataFrame with original data plus classification results
    """
    
    # Get classifications
    classifications = main(target_modality_list, devices)
    
    return classifications 