import os
import json
import base64
from typing import List, Optional, Dict, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import anthropic
from AIClientBase import AIClientBase

import time, logging, random
from MultiClassPrediction import MultiClassImageTask, retry_with_exponential_backoff

class MultiClassPredictionHybrid(AIClientBase):
    """
    A class that combines image and text analysis for classification,
    using both visual features and textual descriptions.
    """
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        batch_size: int = 20,
        max_workers: int = 5,
        use_features: bool = False
    ):
        
        super().__init__(model)
        self.model = model
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.use_features = use_features
        
    
    def update_model(self, model: str):
        """Update the model used for processing"""
        self.model = model
    
    def _create_message_content(
        self,
        batch: List[MultiClassImageTask],
        model_type: str,
        image_media_type: str
    ) -> Tuple[List[Dict], Dict[int, str]]:
        """Create message content combining both image and text analysis"""
        encoded_images = []
        path_mapping = {}
        instruction_details = []
        output_format = {"images": []}
        
        for i, task in enumerate(batch):
            num_pred = task.num_predictions
            path_mapping[i] = [task.image_path, task.encoded_image]
            
            features = task.features
            feature_instruction = (
                f" Consider these features: {', '.join(features)}\n"
                if self.use_features and features else ""
            )
            
            description = task.get_image_textual_description()
            if not description:
                raise ValueError(f"No description for {task.image_path}")
            
            instruction_details.append(
                f"Analysis Task {i}:\n"
                f"Image {i} and its description:\n{description}\n"
                f"Classify into {num_pred} UNIQUE classes from: {', '.join(task.classes)}"
                f"{feature_instruction}"
            )
            
            output_format["images"].append({
                "image_index": i,
                "image_path": task.image_path,
                "predicted_classes": [
                    {
                        "class": f"[class from: {', '.join(task.classes)}]",
                        "confidence": 0.0,
                        "reasoning": "Explanation combining visual and textual evidence",
                        "key_features": ["visual_feature1", "textual_feature1", "visual_feature2", "textual_feature2", "...", "visual_featureN", "textual_featureN"],
                        "combined_evidence": "How visual and textual features support this classification"
                    } for _ in range(num_pred)
                ]
            })
            
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
                    "**Let's think about each medical image step by step** and then analyze each medical case using both the image and its textual description.\n\n"
                    "Case-specific requirements:\n" +
                    "\n".join(instruction_details) +
                    "\n\nProvide JSON output in the following format:\n"
                    f"{json.dumps(output_format, indent=2)}\n\n"
                    "Requirements:\n"
                    "- Consider both visual evidence and textual description\n"
                    "- Each case must have UNIQUE predicted classes (no duplicates)\n"
                    "- Include the image_index in each prediction\n"
                    "- Confidence scores should sum to 1.0 for each case\n"
                    "- Provide separate visual and textual features\n"
                    "- Explain how visual and textual evidence combine to support each prediction\n"
                    "- Use only the specified classes\n"
                    "- Never predict the same class more than once for a single case"
                )
            },
            *encoded_images
        ]
        
        return message_content, path_mapping

    @retry_with_exponential_backoff()
    def _process_openai(
        self,
        batch: List[MultiClassImageTask],
        model_type: str,
        image_media_type: str
    ) -> Dict:
        """Process a batch using both image and text with OpenAI"""
        content, path_mapping = self._create_message_content(batch, model_type, image_media_type)
        
        response = self.openai_client.chat.completions.create(
            model=self.model,  # Use vision model for combined analysis
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical specialist who analyzes both medical images and their descriptions for comprehensive classification."
                },
                {"role": "user", "content": content}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        result = self._add_paths_to_results(result, batch)
        return result

    @retry_with_exponential_backoff()
    def _process_claude(
        self,
        batch: List[MultiClassImageTask],
        model_type: str,
        image_media_type: str
    ) -> Dict:
        """Process a batch using both image and text with Claude"""
        content, path_mapping = self._create_message_content(batch, model_type, image_media_type)
        
        response = self.claude_client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": content}]
        )
        
        result = json.loads(response.content[0].text)
        result = self._add_paths_to_results(result, batch)
        return result

    def _add_paths_to_results(self, results: Dict, batch: List[MultiClassImageTask]) -> Dict:
        """Add paths and descriptions back to the results"""
        if isinstance(results, dict) and "images" in results:
            for image_result in results["images"]:
                if "image_index" in image_result:
                    index = image_result["image_index"]
                    image_result["image_path"] = batch[index].image_path
                    image_result["image_textual_description"] = batch[index].image_textual_description
                    if hasattr(batch[index], 'encoded_image'):
                        image_result["encoded_image"] = batch[index].encoded_image
        return results

    def process_hybrid(
        self,
        tasks: List[MultiClassImageTask],
        model_type: str = "openai",
        image_media_type: str = "image/png",
        parallel: bool = True
    ) -> Dict:
        """
        Process both images and their descriptions in parallel batches.
        
        Args:
            tasks: List of MultiClassImageTask objects with both images and descriptions
            model_type: AI model to use ('openai' or 'claude')
            image_media_type: Image type (default: "image/png")
            parallel: Whether to use parallel processing
            
        Returns:
            dict: Combined predictions using both visual and textual evidence
        """
        batches = [tasks[i:i + self.batch_size] for i in range(0, len(tasks), self.batch_size)]
        process_func = self._process_openai if model_type.lower() == "openai" else self._process_claude
        
        all_results = {"images": []}
        if parallel and len(batches) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        process_func,
                        batch,
                        model_type,
                        image_media_type
                    ): batch for batch in batches
                }
                
                for future in as_completed(futures):
                    try:
                        results = future.result()
                        all_results["images"].extend(results["images"])
                    except Exception as e:
                        batch = futures[future]
                        print(f"Batch processing failed: {e}")
                        print(f"Failed batch: {[task.image_path for task in batch]}")
        else:
            for batch in batches:
                results = process_func(batch, model_type, image_media_type)
                all_results["images"].extend(results["images"])
        
        return all_results