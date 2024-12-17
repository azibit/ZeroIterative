import os
import json
import base64
from typing import List, Optional, Dict, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import anthropic

import time, logging, random
from MultiClassPrediction import MultiClassImageTask, retry_with_exponential_backoff

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from AIClientBase import AIClientBase

class MultiClassDescription(AIClientBase):
    """
    A class to handle parallel batch image processing for generating structured medical descriptions,
    supporting detailed anatomical and radiological descriptions.
    """
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        batch_size: int = 20,
        max_workers: int = 5
    ):
        
        super().__init__(model)
        self.model = model
        self.batch_size = batch_size
        self.max_workers = max_workers

    def update_model(self, model: str):
        """Update the model used for processing"""
        self.model = model

    @retry_with_exponential_backoff()
    def _process_openai(
        self,
        batch: List[MultiClassImageTask],
        model_type: str,
        image_media_type: str
    ) -> Dict:
        """Process a batch of images with OpenAI"""
        content, path_mapping = self._create_message_content(batch, model_type, image_media_type)
        
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a medical imaging specialist focused on detailed anatomical and radiological descriptions."},
                {"role": "user", "content": content}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        result = self._add_image_paths_to_results(result, batch)
        return result

    @retry_with_exponential_backoff()
    def _process_claude(
        self,
        batch: List[MultiClassImageTask],
        model_type: str,
        image_media_type: str
    ) -> Dict:
        """Process a batch of images with Claude"""
        content, path_mapping = self._create_message_content(batch, model_type, image_media_type)
        
        response = self.claude_client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0,
            messages=[
                {"role": "user", "content": content}
            ]
        )
        
        result = json.loads(response.content[0].text)
        result = self._add_image_paths_to_results(result, batch)
        return result

    def _create_message_content(
        self,
        batch: List[MultiClassImageTask],
        model_type: str,
        image_media_type: str
    ) -> Tuple[List[Dict], Dict[int, str]]:
        """Create message content for API calls to generate structured medical descriptions"""
        
        encoded_images = []
        path_mapping = {}
        output_format = {
            "images": []
        }
        
        for i, task in enumerate(batch):
            # Perform the path mapping
            path_mapping[i] = [task.image_path, task.encoded_image]

            # Form the output format using medical reporting structure
            output_format_per_image = {
                "image_index": i,
                "image_path": task.image_path,
                "technical_quality": {
                    "image_modality": "",
                    "positioning": "",
                    "image_quality": ""
                },
                "anatomical_description": {
                    "location": "",
                    "anatomical_landmarks": [],
                    "orientation": ""
                },
                "findings": {
                    "composition": {
                        "density_characteristics": "",
                        "internal_architecture": ""
                    },
                    "morphology": {
                        "shape": "",
                        "margins": "",
                        "size": ""
                    },
                    "distribution": {
                        "spatial_arrangement": "",
                        "relationship_to_landmarks": ""
                    },
                    "signal_characteristics": {
                        "intensity_patterns": "",
                        "texture": ""
                    }
                }
            }
            output_format["images"].append(output_format_per_image)

            # Handle image encoding
            encoded = task.encoded_image
            if model_type.lower() == "openai":
                encoded_images.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_media_type};base64,{encoded}"
                    }
                })
            else:  # Claude format
                encoded_images.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": encoded
                    }
                })
        
        message_content = [
            {
                "type": "text",
                "text": (
                    "Provide a structured medical description of each image following standard radiological "
                    "reporting principles. Maintain objectivity and focus on observable characteristics.\n\n"
                    "For each image, describe:\n\n"
                    "1. TECHNICAL ASSESSMENT:\n"
                    "   - Imaging modality and technique used\n"
                    "   - Patient positioning and orientation\n"
                    "   - Image quality and technical adequacy\n\n"
                    "2. ANATOMICAL LOCALIZATION:\n"
                    "   - Precise anatomical region and boundaries\n"
                    "   - Key anatomical landmarks visible\n"
                    "   - Anatomical planes and directions\n\n"
                    "3. DETAILED FINDINGS:\n"
                    "   a) Composition:\n"
                    "      - Density/intensity characteristics\n"
                    "      - Internal architectural patterns\n"
                    "   b) Morphology:\n"
                    "      - Shape characteristics\n"
                    "      - Border/margin features\n"
                    "      - Size and proportions\n"
                    "   c) Distribution:\n"
                    "      - Spatial arrangement\n"
                    "      - Relationship to anatomical landmarks\n"
                    "   d) Signal Characteristics:\n"
                    "      - Pattern of intensity variations\n"
                    "      - Texture analysis\n\n"
                    "Requirements:\n"
                    "- Use standard medical imaging terminology\n"
                    "- Follow systematic approach to description\n"
                    "- Maintain purely objective observations\n"
                    "- Include quantitative measurements where applicable\n"
                    "- Use anatomical directions and planes\n"
                    "- Avoid diagnostic interpretations or classifications\n"
                    "- Be specific about location and distribution\n\n"
                    f"Provide JSON output in the following format:\n{json.dumps(output_format, indent=2)}"
                )
            },
            *encoded_images
        ]
        
        return message_content, path_mapping

    def _add_image_paths_to_results(self, results: Dict, batch: List[MultiClassImageTask]):
        """Add image paths back to the results using indices"""
        if isinstance(results, dict) and "images" in results:
            for image_result in results["images"]:
                if "image_index" in image_result:
                    index = image_result["image_index"]
                    image_result["image_path"] = batch[index].image_path
                    image_result["encoded_image"] = batch[index].encoded_image
        return results

    def process_images(
        self,
        image_tasks: Union[List[Tuple[str, List[str]]], List[MultiClassImageTask]],
        model_type: str = "openai",
        image_media_type: str = "image/png",
        parallel: bool = True
    ) -> Dict:
        """
        Process images in parallel batches to generate structured medical descriptions.
        
        Args:
            image_tasks: List of MultiClassImageTask objects or tuples of (image_path, classes_list)
            model_type: AI model to use ('openai' or 'claude')
            image_media_type: Image type (default: "image/png")
            parallel: Whether to use parallel processing (default: True)
            
        Returns:
            dict: Combined medical descriptions for all images with their paths
        """
        # Convert tuples to MultiClassImageTask objects if necessary
        tasks = [
            task if isinstance(task, MultiClassImageTask) else MultiClassImageTask(task[0], task[1])
            for task in image_tasks
        ]
        
        # Create batches
        batches = [
            tasks[i:i + self.batch_size] 
            for i in range(0, len(tasks), self.batch_size)
        ]
        
        # Select processing function
        process_func = self._process_openai if model_type.lower() == "openai" else self._process_claude
        if model_type.lower() not in ["openai", "claude"]:
            raise ValueError(f"Unsupported model: {model_type}. Use 'openai' or 'claude'")
        
        # Process batches
        all_results = {"images": []}
        if parallel and len(batches) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_batch = {
                    executor.submit(process_func, batch, model_type, image_media_type): batch 
                    for batch in batches
                }
                
                for future in as_completed(future_to_batch):
                    try:
                        results = future.result()
                        all_results["images"].extend(results["images"])
                    except Exception as e:
                        batch = future_to_batch[future]
                        print(f"Batch processing failed: {e}")
                        print(f"Failed batch: {[task.image_path for task in batch]}")
        else:
            # Sequential processing
            for batch in batches:
                results = process_func(batch, model_type, image_media_type)
                all_results["images"].extend(results["images"])
        
        return all_results