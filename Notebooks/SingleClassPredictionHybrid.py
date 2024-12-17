import os
import json
import base64
from typing import List, Optional, Dict, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from openai import OpenAI
import anthropic
from SingleClassPrediction import SingleClassImageTask
from AIClientBase import AIClientBase

class SingleClassPredictionHybrid(AIClientBase):
    """
    A class that combines image and text analysis for single-class prediction,
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
        
        # # Initialize clients
        # self.claude_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_KEY')) 
        # self.openai_client = OpenAI(api_key=os.getenv('OPENAI_KEY'))
    
    def update_model(self, model: str):
        """Update the model used for processing"""
        self.model = model
    
    def _create_instruction_message(
        self,
        batch: List[SingleClassImageTask],
        image_media_type: str,
        model: str
    ) -> Tuple[Dict, Dict[str, str], List[Dict]]:
        """Create combined instruction message using both image and text"""
        path_mapping = {f"image_{i+1}": task.image_path for i, task in enumerate(batch)}
        
        output_format = []
        instruction_details = []
        image_messages = []
        
        for i, task in enumerate(batch):
            # Create output format
            output_format.append(
                f'"image_{i+1}": ' + 
                '{\n' +
                '    "predicted_classes": [\n' +
                '        {\n' +
                '            "class": "[class name from: ' + ', '.join(task.classes) + ']",\n' +
                '            "confidence": 0.0,\n' +
                '            "key_features": ["visual_feature1", "textual_feature1", "visual_feature2", "textual_feature2", "...", "visual_featureN", "textual_featureN"]\n' +
                '            "combined_reasoning": "Explanation combining visual and textual evidence"\n' +
                '        }\n' +
                '    ]\n' +
                '}'
            )
            
            # Get features and description
            features = task.features
            feature_instruction = (
                f" Consider these features: {', '.join(features)}\n"
                if self.use_features and features else ""
            )
            
            description = task.image_textual_description
            if not description:
                raise ValueError(f"No description for {task.image_path}")
            
            # Create instruction details
            instruction_details.append(
                f"Case {i+1}:\nImage {i+1} and its description:\n{description}\n"
                f"Classify using classes: {', '.join(task.classes)}{feature_instruction}"
            )
            
            # Create image message
            with open(task.image_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode('utf-8')
                if model.lower() == "openai":
                    image_messages.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_media_type};base64,{encoded}"
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
        
        instruction_text = {
            "type": "text",
            "text": (
                "**Let's think about each medical image step by step** and then classify it according to its specified categories based on visible features.\n\n"
                "Cases to analyze:\n" +
                "\n".join(instruction_details) +
                "\n\nOUTPUT FORMAT:\n"
                "{\n" +
                ",\n".join(output_format) +
                "\n}\n\n"
                "REQUIREMENTS:\n"
                "- Analyze both visual features and textual descriptions\n"
                "- Confidence scores should sum to 1.0 for each case\n"
                "- List visual evidence observed in images\n"
                "- List textual evidence from descriptions\n"
                "- Provide combined reasoning explaining classification\n"
                f"- Analyze all {len(batch)} cases\n"
                "- Each prediction must include confidence score and evidence\n"
                "- Consider both image and text equally in analysis\n\n"
                "Respond only with the JSON object."
            )
        }
        
        return instruction_text, path_mapping, image_messages

    def _process_batch(
        self,
        batch: List[SingleClassImageTask],
        model: str,
        image_media_type: str
    ) -> List[Dict]:
        """Process a single batch using both image and text analysis"""
        instruction_msg, path_mapping, image_msgs = self._create_instruction_message(
            batch, image_media_type, model
        )
        
        if model.lower() == "openai":
            raw_results = self._process_openai(instruction_msg, image_msgs)
        elif model.lower() == "claude":
            raw_results = self._process_claude(instruction_msg, image_msgs)
        else:
            raise ValueError(f"Unsupported model: {model}. Use 'openai' or 'claude'")
        
        # Process results
        processed_results = []
        for i, task in enumerate(batch):
            result = raw_results[f"image_{i+1}"]
            result["image_path"] = path_mapping[f"image_{i+1}"]
            result["image_textual_description"] = task.image_textual_description
            processed_results.append(result)
        
        return processed_results

    def _process_openai(
        self,
        instruction_msg: Dict,
        image_msgs: List[Dict]
    ) -> Dict:
        """Process with OpenAI"""
        messages = [
            {
                "role": "system",
                "content": "You are a medical specialist who analyzes both medical images and their descriptions for comprehensive classification."
            },
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
        """Process with Claude"""
        formatted_content = [instruction_msg, *image_msgs]
        
        response = self.claude_client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": formatted_content}]
        )
        
        return json.loads(response.content[0].text)

    def process_hybrid(
        self,
        tasks: List[SingleClassImageTask],
        model: str = "openai",
        image_media_type: str = "image/png",
        parallel: bool = True
    ) -> Dict:
        """
        Process cases using both image and text analysis.
        
        Args:
            tasks: List of SingleClassImageTask objects with both images and descriptions
            model: AI model to use ('openai' or 'claude')
            image_media_type: Image type (default: "image/png")
            parallel: Whether to use parallel processing
            
        Returns:
            dict: Combined predictions using both visual and textual evidence
        """
        batches = [tasks[i:i + self.batch_size] for i in range(0, len(tasks), self.batch_size)]
        
        all_results = {"images": []}
        if parallel and len(batches) > 1:
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
                        all_results["images"].extend(results)
                    except Exception as e:
                        batch = future_to_batch[future]
                        print(f"Batch processing failed: {e}")
                        print(f"Failed batch: {[task.image_path for task in batch]}")
        else:
            for batch in batches:
                results = self._process_batch(batch, model, image_media_type)
                all_results["images"].extend(results)
        
        return all_results