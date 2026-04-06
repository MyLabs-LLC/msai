# Test script for AugmentedPromptAgent class

from workflow_agents.base_agents import AugmentedPromptAgent
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


openai_api_key = load_openai_api_key()

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

augmented_agent = AugmentedPromptAgent(openai_api_key, persona)
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
logger.info(augmented_agent_response)

# Knowledge source: The agent used the LLM's training data to answer the question.
# The system prompt specifying the persona caused the agent to adopt the role of a
# college professor, changing the tone and format of the response (e.g., starting
# with "Dear students,"). The factual content still comes from the model's training data.
