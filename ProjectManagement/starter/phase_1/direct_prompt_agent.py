# Test script for DirectPromptAgent class

from workflow_agents.base_agents import DirectPromptAgent
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

prompt = "What is the Capital of France?"

direct_agent = DirectPromptAgent(openai_api_key)
direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent
logger.info(direct_agent_response)

# The DirectPromptAgent uses the LLM's training data as its knowledge source.
# No system prompt or external knowledge is provided, so the agent relies entirely
# on the information learned during the model's pre-training phase.
logger.info("Knowledge source: The agent used the LLM's training data (pre-trained knowledge) to answer the question. No external context or system prompt was provided.")
