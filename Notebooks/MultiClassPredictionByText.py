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

class MultiClassPredictionByText(AIClientBase):
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
    ) -> Tuple[List[Dict], Dict[int, str]]:
        path_mapping = {}
        instruction_details = []
        output_format = {"images": []}
        
        for i, task in enumerate(batch):
            num_pred = task.num_predictions
            path_mapping[i] = task.image_path
            
            features = task.features
            feature_instruction = f" Consider these features: {', '.join(features)}\n" if self.use_features and features else ""
            
            description = task.get_image_textual_description()
            if not description:
                raise ValueError(f"No description for {task.image_path}")
            
            instruction_details.append(
                f"Description {i}:\n{description}\n"
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
                        "reasoning": "Explanation for this classification",
                        "key_features": ["feature1", "feature2", "...", "featureN"]
                    } for _ in range(num_pred)
                ]
            })
        
        message_content = [{
            "type": "text",
            "text": (
                "**Let's think about each medical image step by step** and then classify it according to its specified categories based on visible features.\n\n"
                "Descriptions and requirements:\n" +
                "\n".join(instruction_details) +
                "\n\nProvide JSON output in the following format:\n"
                f"{json.dumps(output_format, indent=2)}\n\n"
                "Requirements:\n"
                "- Each description must have UNIQUE predicted classes (no duplicates)\n"
                "- Include the image_index in each prediction\n"
                "- Confidence scores should sum to 1.0 for each description\n"
                "- Provide detailed reasoning for each classification\n"
                "- List key features that support each classification\n"
                "- Use only the specified classes\n"
                "- Base classifications solely on the provided descriptions\n"
                "- Never predict the same class more than once for a single description"
            )
        }]
        
        return message_content, path_mapping

    @retry_with_exponential_backoff()
    def _process_openai(self, batch: List[MultiClassImageTask]) -> Dict:
        content, path_mapping = self._create_message_content(batch)
        
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a medical specialist who classifies cases based on detailed medical descriptions."},
                {"role": "user", "content": content}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        result = self._add_paths_to_results(result, batch)
        return result

    @retry_with_exponential_backoff()
    def _process_claude(self, batch: List[MultiClassImageTask]) -> Dict:
        content, path_mapping = self._create_message_content(batch)
        
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
        """Add paths back to the results using indices"""
        if isinstance(results, dict) and "images" in results:
            for image_result in results["images"]:
                if "image_index" in image_result:
                    index = image_result["image_index"]
                    image_result["image_path"] = batch[index].image_path
                    if hasattr(batch[index], 'image_textual_description'):
                        image_result["image_textual_description"] = batch[index].image_textual_description
        return results

    def process_descriptions(
        self,
        tasks: List[MultiClassImageTask],
        model_type: str = "openai",
        parallel: bool = True
    ) -> Dict:        
        batches = [tasks[i:i + self.batch_size] for i in range(0, len(tasks), self.batch_size)]
        process_func = self._process_openai if model_type.lower() == "openai" else self._process_claude

        print("Inside process Descriptions for MultiClassPredictionByText: ", batches)
        
        all_results = {"images": []}
        if parallel and len(batches) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(process_func, batch): batch for batch in batches}
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
                results = process_func(batch)
                all_results["images"].extend(results["images"])
        
        return all_results