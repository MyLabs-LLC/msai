# Test script for RoutingAgent class

from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent
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

persona = "You are a college professor"

knowledge = "You know everything about Texas"
texas_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

knowledge = "You know everything about Europe"
europe_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

persona = "You are a college math professor"
knowledge = "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"
math_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

routing_agent = RoutingAgent(openai_api_key)
agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        "func": lambda x: texas_agent.respond(x)
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        "func": lambda x: europe_agent.respond(x)
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        "func": lambda x: math_agent.respond(x)
    }
]

routing_agent.agents = agents

logger.info("Prompt 1: Tell me about the history of Rome, Texas")
response1 = routing_agent.route("Tell me about the history of Rome, Texas")
logger.info(f"Response:\n{response1}\n")

logger.info("Prompt 2: Tell me about the history of Rome, Italy")
response2 = routing_agent.route("Tell me about the history of Rome, Italy")
logger.info(f"Response:\n{response2}\n")

logger.info("Prompt 3: One story takes 2 days, and there are 20 stories")
response3 = routing_agent.route("One story takes 2 days, and there are 20 stories")
logger.info(f"Response:\n{response3}\n")
