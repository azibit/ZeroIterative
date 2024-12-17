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

def retry_with_exponential_backoff(
    initial_delay: float = 1,
    max_delay: float = 60,
    max_retries: int = 3,
    backoff_factor: float = 2,
    jitter: bool = True
):
    """
    Decorator for retrying a function with exponential backoff.
    
    Args:
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        max_retries: Maximum number of retries
        backoff_factor: Multiplicative factor for exponential backoff
        jitter: Whether to add random jitter to delay
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            retries = 0
            
            while True:
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        logging.error(f"Max retries ({max_retries}) exceeded: {str(e)}")
                        raise
                    
                    # Calculate delay with optional jitter
                    current_delay = min(delay * (backoff_factor ** (retries - 1)), max_delay)
                    if jitter:
                        current_delay = current_delay * (1 + random.uniform(-0.1, 0.1))
                    
                    logging.warning(
                        f"OpenAI API error: {str(e)}. "
                        f"Retrying in {current_delay:.2f} seconds... "
                        f"(Attempt {retries}/{max_retries})"
                    )
                    
                    time.sleep(current_delay)
                
                except Exception as e:
                    logging.error(f"Unexpected error: {str(e)}")
                    raise
                    
        return wrapper
    return decorator

@dataclass
class MultiClassImageTask:
    """Data class to hold image path and its associated classes"""
    image_path: str
    classes: List[str]
    num_predictions: Optional[int] = None
    encoded_image: Optional[str] = None
    features: Optional[List[str]] = None
    image_textual_description: Optional[str] = None

    def get_image_path(self) -> str:
        """Get the image path"""
        return self.image_path
    
    def get_classes(self) -> List[str]:
        """Get the classes"""
        return self.classes
    
    def get_num_predictions(self) -> Optional[int]:
        """Get the number of predictions"""
        return self.num_predictions
    
    def get_features(self) -> Optional[List[str]]:
        """Get the features"""
        return self.features
    
    def get_encoded_image(self) -> Optional[str]:
        """Get the encoded image"""
        return self.encoded_image
    
    def set_encoded_image(self, encoded_image: str):
        """Set the encoded image"""
        self.encoded_image = encoded_image
    
    def set_num_predictions(self, num_predictions: int):
        """Set the number of predictions"""
        self.num_predictions = num_predictions

    def set_classes(self, classes: List[str]):
        """Set the classes"""
        self.classes = classes

    def set_features(self, features: List[str]):
        """Set the features"""
        self.features = features

    def get_image_textual_description(self) -> Optional[str]:
        """Get the image textual description"""
        return self.image_textual_description
    
    def set_image_textual_description(self, image_textual_description: str):
        """Set the image textual description"""
        self.image_textual_description = image_textual_description

class MultiClassPrediction(AIClientBase):
    """
    A class to handle parallel batch image processing using different AI models,
    supporting different classes per image.
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

    def update_use_features(self, use_features: bool):
        """Update the feature usage flag"""
        self.use_features = use_features

    def process_images(
        self,
        image_tasks: Union[List[Tuple[str, List[str]]], List[MultiClassImageTask]],
        model_type: str = "openai",
        image_media_type: str = "image/png",
        parallel: bool = True
    ) -> Dict:
        """
        Process images in parallel batches using specified AI model.
        
        Args:
            image_tasks: List of MultiClassImageTask objects or tuples of (image_path, classes_list)
            model: AI model to use ('openai' or 'claude')
            image_media_type: Image type (default: "image/png")
            parallel: Whether to use parallel processing (default: True)
            
        Returns:
            dict: Combined predictions for all images with their paths
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
            batch_results = self._process_parallel(
                batches,
                process_func,
                model_type,
                image_media_type
            )
            all_results["images"].extend(batch_results)
        else:
            # Sequential processing
            for batch in batches:
                results = process_func(
                    batch,
                    model_type,
                    image_media_type
                )
                all_results["images"].extend(results["images"])
        
        return all_results

    def _process_parallel(
        self,
        batches: List[List[MultiClassImageTask]],
        process_func,
        model_type: str,
        image_media_type: str
    ) -> List[Dict]:
        """Process batches in parallel using threads"""
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches to thread pool
            future_to_batch = {
                executor.submit(
                    process_func,
                    batch,
                    model_type,
                    image_media_type
                ): batch 
                for batch in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                try:
                    results = future.result()
                    all_results.extend(results["images"])
                except Exception as e:
                    batch = future_to_batch[future]
                    print(f"Batch processing failed: {e}")
                    print(f"Failed batch: {[task.image_path for task in batch]}")
        
        return all_results

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
                {"role": "system", "content": "You are a medical image classifier."},
                {"role": "user", "content": content}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        # Parse response and add image paths
        result = json.loads(response.choices[0].message.content)

        # Add paths back to the response using indices
        result = self._add_image_paths_to_results(result, batch)
                
        return result
    
    def _add_image_paths_to_results(self, results: Dict, batch: List[MultiClassImageTask]):
        """Add image paths back to the results using indices"""
        if isinstance(results, dict) and "images" in results:
            for image_result in results["images"]:
                if "image_index" in image_result:
                    index = image_result["image_index"]
                    image_result["image_path"] = batch[index].image_path
                    image_result["encoded_image"] = batch[index].encoded_image
                    # del image_result["image_index"]  # Clean up the index field
        return results

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
            messages=[{"role": "user", "content": content}]
        )        
        # Parse response and add image paths
        result = json.loads(response.content[0].text)
        
        # Add paths back to the response using indices
        result = self._add_image_paths_to_results(result, batch)
                
        return result

    def _create_message_content(
            self,
            batch: List[MultiClassImageTask],
            model_type: str,
            image_media_type: str
        ) -> Tuple[List[Dict], Dict[int, str]]:  # Simple index to path mapping
            """Create message content for API calls with per-image classes"""
            
            encoded_images = []
            path_mapping = {}
            instruction_details = []
            output_format = {}
            output_format["images"] = []
            
            for i, task in enumerate(batch):
                num_pred = task.num_predictions #min(task.num_predictions or len(task.classes), len(task.classes))

                # Perform the path mapping
                path_mapping[i] = [task.image_path, task.encoded_image]

                # Get the features to use for this task
                features = task.features
                feature_instruction_steps = ""
                if self.use_features and features:
                    feature_instruction_steps = f" Using the following features: {', '.join(features)}\n"

                # Form the instruction details
                instruction_details.append(
                    f"Image {i}: Predict top {num_pred} UNIQUE classes from: {', '.join(task.classes)} {feature_instruction_steps}"
                )

                # Form the output format
                output_format_per_image = {
                        "image_index": i,  # Add explicit index for matching
                        "predicted_classes": [
                            {
                                "class": f"[class from: {', '.join(task.classes)}]",
                                "confidence": 0.0,
                                "key_features": ["feature1", "feature2", "...", "featureN"]
                            } for _ in range(num_pred)
                        ]
                    }
                output_format["images"].append(output_format_per_image)

                # Form the base64 encoded images
                encoded = task.encoded_image
                if model_type.lower() == "openai":
                    encoded_images.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_media_type};base64,{encoded}",
                            # "detail": "high"
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
                        "**Let's think about each medical image step by step** and then analyze each medical image according to its specific classification requirements.\n\n"
                        "Image-specific requirements:\n" +
                        "\n".join(instruction_details) +
                        "\n\nProvide JSON output in the following format:\n"
                        f"{json.dumps(output_format, indent=2)}\n\n"
                        "Requirements:\n"
                        "- Each image must have UNIQUE predicted classes (no duplicates)\n"
                        "- Include the image_index in each prediction\n"
                        "- Confidence scores should sum to 1.0 for each image\n"
                        "- List only visible features and its location in the image\n"
                        # And its location in the image\n"
                        "- Provide detailed key features for each prediction\n"
                        "- Use only the specified classes for each image\n"
                        "- Never predict the same class more than once for a single image"
                    )
                },
                *encoded_images
            ]
            
            return message_content, path_mapping