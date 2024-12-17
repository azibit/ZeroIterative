import os
import json
import base64
from typing import List, Optional, Dict, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from openai import OpenAI
import anthropic
import os
from openai import AzureOpenAI
from AIClientBase import AIClientBase

@dataclass
class SingleClassImageTask:
    """Data class to hold image path and its associated classes"""
    image_path: str
    classes: List[str]
    features: Optional[List[str]] = None
    image_textual_description: Optional[str] = None

class SingleClassPrediction(AIClientBase):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        batch_size: int = 20,
        max_workers: int = 5,  # Default number of parallel threads,
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

    def process_images(
        self,
        image_tasks: Union[List[Tuple[str, List[str]]], List[SingleClassImageTask]],
        model: str = "openai",
        image_media_type: str = "image/png",
        parallel: bool = True
    ) -> Dict:
        """Process multiple images with different classification classes"""
        # Convert tuples to SingleClassImageTask objects if necessary
        tasks = [
            task if isinstance(task, SingleClassImageTask) else SingleClassImageTask(*task)
            for task in image_tasks
        ]
        
        # Create batches
        batches = [
            tasks[i:i + self.batch_size] 
            for i in range(0, len(tasks), self.batch_size)
        ]
        
        # Process batches
        if parallel and len(batches) > 1:
            batch_results = self._process_parallel(
                batches,
                model,
                image_media_type
            )
        else:
            batch_results = []
            for batch in batches:
                results = self._process_batch(
                    batch,
                    model,
                    image_media_type
                )
                batch_results.extend(results)
        
        # Return results
        result_dict = {
            "images": batch_results
        }
        
        return result_dict

    def _process_parallel(
        self,
        batches: List[List[SingleClassImageTask]],
        model: str,
        image_media_type: str
    ) -> List[Dict]:
        """Process batches in parallel using threads"""
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(
                    self._process_batch,
                    batch,
                    model,
                    image_media_type
                ): batch 
                for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    batch = future_to_batch[future]
                    print(f"Batch processing failed: {e}")
                    print(f"Failed batch: {[task.image_path for task in batch]}")
        
        return all_results

    def _process_batch(
        self,
        batch: List[SingleClassImageTask],
        model: str,
        image_media_type: str
    ) -> List[Dict]:
        """Process a single batch of image tasks"""
        instruction_msg, path_mapping = self._create_instruction_message(batch)
        image_msgs = self._create_image_messages(batch, image_media_type, model)
        
        if model.lower() == "openai":
            raw_results = self._process_openai(instruction_msg, image_msgs)
        elif model.lower() == "claude":
            raw_results = self._process_claude(instruction_msg, image_msgs)
        else:
            raise ValueError(f"Unsupported model: {model}. Use 'openai' or 'claude'")
        
        # Add image paths to results
        processed_results = []
        for i, task in enumerate(batch):
            image_result = raw_results[f"image_{i+1}"]
            image_result["image_path"] = path_mapping[f"image_{i+1}"]
            processed_results.append(image_result)
        
        return processed_results

    def _create_instruction_message(
            self,
            batch: List[SingleClassImageTask]
        ) -> Tuple[Dict, Dict[str, str]]:  # Updated return type to include path mapping
            """
            Create the instruction message with format specifications for each image.
            Returns a tuple of (instruction_message, path_mapping) where path_mapping
            maps image_ids to their paths for later reference.
            """
            # Create a mapping of image IDs to paths for tracking
            path_mapping = {f"image_{i+1}": task.image_path for i, task in enumerate(batch)}
            
            output_format = []
            for i, task in enumerate(batch):
                output_format.append(
                    f'"image_{i+1}": ' + 
                    '{\n' +
                    '    "predicted_classes": [\n' +
                    '        {\n' +
                    '            "class": "[class name from: ' + ', '.join(task.classes) + ']",\n' +
                    '            "confidence": 0.0,\n' +
                    '            "key_features": ["feature1", "feature2", ...]\n' +
                    '        }\n' +
                    '    ]\n' +
                    '}'
                )


            
            # Create instruction details without paths
            instruction_details = []
            for i, task in enumerate(batch):

                # Get the features
                features = task.features
                feature_instruction_steps = ""
                if self.use_features and features:
                    feature_instruction_steps = f" and using the following features: {', '.join(features)}\n"

                instruction_details.append(
                    f"Image {i+1}: Classify using classes: {', '.join(task.classes)} {feature_instruction_steps}"
                )
            
            instruction_text = (
                "You are a medical image classification system. Your output is always a JSON object containing visual analysis for multiple images. "
                "**Let's think about each medical image step by step** and then classify it according to its specified categories based on visible features.\n\n"
                "Images to analyze:\n" +
                "\n".join(instruction_details) +
                "\n\nOUTPUT FORMAT:\n"
                "{\n" +
                ",\n".join(output_format) +
                "\n}\n\n"
                "REQUIREMENTS:\n"
                "- List only visible features and their locations in the image\n"
                "- Include shape, color, texture, borders\n"
                "- Confidence scores should sum to 1.0 for each image\n"
                "- Focus on distinguishing patterns\n"
                f"- Analyze all {len(batch)} images\n"
                "- Each class prediction must include confidence score and key features\n"
                "- Maintain consistent order with input images\n\n"
                "Respond only with the JSON object."
            )
            
            return {
                "type": "text",
                "text": instruction_text
            }, path_mapping

    def _create_image_messages(
        self,
        batch: List[SingleClassImageTask],
        image_media_type: str,
        model: str
    ) -> List[Dict]:
        """Create the image messages with base64 encoding"""
        image_messages = []
        for task in batch:
            with open(task.image_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode('utf-8')
                if model.lower() == "openai":
                    image_messages.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_media_type};base64,{encoded}",
                            # "detail": "high"
                        }
                    })
                else:  # Claude format
                    image_messages.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": encoded
                        }
                    })
        return image_messages

    def _process_openai(
        self,
        instruction_msg: Dict,
        image_msgs: List[Dict]
    ) -> Dict:
        """Process messages with OpenAI"""
        messages = [
            {"role": "system", "content": "You are a medical image classifier."},
            {"role": "user", "content": [instruction_msg, *image_msgs]}
        ]

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)

    def _process_claude(
        self,
        instruction_msg: Dict,
        image_msgs: List[Dict]
    ) -> Dict:
        """Process messages with Claude"""
        # Combine instruction and images into Claude's format
        formatted_content = [instruction_msg, *image_msgs]

        print("Instruction Messages:", instruction_msg)
        
        response = self.claude_client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0,
            messages=[{
                "role": "user",
                "content": formatted_content
            }]
        )
        
        return json.loads(response.content[0].text)