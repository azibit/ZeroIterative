import os
from openai import AzureOpenAI, OpenAI
import anthropic

class AIClientBase:
    def __init__(self, model=None):
        self.model = model
        self._initialize_clients()
    
    def _initialize_clients(self):
        # Initialize Anthropic client
        self.claude_client = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_KEY')
        )

        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_PHD_PROJ_KEY1'))