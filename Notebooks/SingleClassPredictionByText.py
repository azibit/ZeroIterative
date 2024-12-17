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

class SingleClassPredictionByText(AIClientBase):
    def __init__(self, model: str = "gpt-4o-mini", batch_size: int = 20, max_workers: int = 5, use_features: bool = False):

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

    def _create_instruction_message(self, batch: List[SingleClassImageTask]) -> Tuple[Dict, Dict[str, str]]:
        path_mapping = {f"image_{i+1}": task.image_path for i, task in enumerate(batch)}
        
        output_format = []
        instruction_details = []
        
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
            
            features = task.features
            feature_instruction = f" and using features: {', '.join(features)}\n" if self.use_features and features else ""
            
            description = task.image_textual_description
            if not description:
                raise ValueError(f"No description provided for case {task.image_path}")
                
            instruction_details.append(
                f"Case {i+1} Description:\n{description}\n"
                f"Classify using classes: {', '.join(task.classes)}{feature_instruction}"
            )

        instruction_text = {
            "type": "text",
            "text": (
                "**Let's think about each medical image step by step** and then classify each medical case description according to specified categories.\n\n"
                "Cases to analyze:\n" +
                "\n".join(instruction_details) +
                "\n\nOUTPUT FORMAT:\n"
                "{\n" +
                ",\n".join(output_format) +
                "\n}\n\n"
                "REQUIREMENTS:\n"
                "- Base classifications solely on provided descriptions\n"
                "- Confidence scores should sum to 1.0 for each case\n"
                "- List key features that support classification\n"
                "- Analyze all cases\n"
                "- Provide confidence score and key features for each prediction\n\n"
                "Respond only with the JSON object."
            )
        }
        
        return instruction_text["text"], path_mapping

    def process_descriptions(
        self,
        tasks: List[SingleClassImageTask],
        model_type: str = "openai",
        parallel: bool = True
    ) -> Dict:
        
        batches = [tasks[i:i + self.batch_size] for i in range(0, len(tasks), self.batch_size)]
        
        all_results = {"images": []}
        process_func = self._process_openai if model_type.lower() == "openai" else self._process_claude
        
        if parallel and len(batches) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(process_func, batch): batch for batch in batches}
                for future in as_completed(futures):
                    try:
                        results = future.result()
                        processed_results = self._process_results(results, futures[future])
                        all_results["images"].extend(processed_results)
                    except Exception as e:
                        print(f"Batch processing failed: {e}")
        else:
            for batch in batches:
                results = process_func(batch)
                processed_results = self._process_results(results, batch)
                all_results["images"].extend(processed_results)
        
        return all_results

    def _process_openai(self, batch: List[SingleClassImageTask]) -> Dict:
        instruction_msg, _ = self._create_instruction_message(batch)
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a medical case classifier."},
                {"role": "user", "content": instruction_msg}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def _process_claude(self, batch: List[SingleClassImageTask]) -> Dict:
        instruction_msg, _ = self._create_instruction_message(batch)
        response = self.claude_client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": instruction_msg}]
        )

        return json.loads(response.content[0].text)

    def _process_results(self, results: Dict, batch: List[SingleClassImageTask]) -> List[Dict]:
        processed_results = []
        for i, task in enumerate(batch):
            result = results[f"image_{i+1}"]
            result["image_path"] = task.image_path
            processed_results.append(result)
        return processed_results