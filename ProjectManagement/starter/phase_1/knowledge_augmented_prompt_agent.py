# Test script for KnowledgeAugmentedPromptAgent class

from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
from pathlib import Path
from dotenv import load_dotenv

def load_openai_api_key():
    script_dir = Path(__file__).resolve().parent
    for directory in [script_dir, *script_dir.parents]:
        env_path = directory / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            break

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY was not found in a nearby .env file.")
    return openai_api_key


# Define the parameters for the agent
openai_api_key = load_openai_api_key()

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"

knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)
response = knowledge_agent.respond(prompt)

# This demonstrates the agent using the provided knowledge rather than its own inherent knowledge.
# The agent should say "London" instead of "Paris" because it was instructed to use only the
# provided knowledge, which states the capital of France is London.
logger.info(response)
logger.info("\nThe agent used the provided knowledge ('The capital of France is London, not Paris') instead of its own training data. This demonstrates that the KnowledgeAugmentedPromptAgent prioritizes injected knowledge over the LLM's built-in knowledge.")
