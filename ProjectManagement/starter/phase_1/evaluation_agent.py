# Test script for EvaluationAgent class

from workflow_agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent
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

# Parameters for the Knowledge Agent
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"
knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

# Parameters for the Evaluation Agent
persona = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."
evaluation_agent = EvaluationAgent(openai_api_key, persona, evaluation_criteria, knowledge_agent, 10)

# Evaluate the prompt and print the response
result = evaluation_agent.evaluate(prompt)
logger.info("\n--- Final Result ---")
logger.info(f"Final Response: {result['final_response']}")
logger.info(f"Evaluation: {result['evaluation']}")
logger.info(f"Iterations: {result['iterations']}")
