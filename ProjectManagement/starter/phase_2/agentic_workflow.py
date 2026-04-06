# agentic_workflow.py

# TODO: 1 - Import agents
from workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
    DualModelQueryEngine,
)

import os
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

# TODO: 2 - Load the OpenAI key into a variable called openai_api_key
def load_api_keys():
    """Search parent directories for .env and load both API keys."""
    script_dir = Path(__file__).resolve().parent
    for directory in [script_dir, *script_dir.parents]:
        env_path = directory / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            break

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY was not found in a nearby .env file.")

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not found — dual-model mode disabled. "
            "Complex agents will use single-model (gpt-3.5-turbo) only."
        )

    return openai_api_key, anthropic_api_key


openai_api_key, anthropic_api_key = load_api_keys()
phase_dir = Path(__file__).resolve().parent

# Build the dual-model engine when both keys are available
dual_engine: Optional[DualModelQueryEngine] = None
if anthropic_api_key:
    dual_engine = DualModelQueryEngine(
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        openai_model="gpt-5.4",
        anthropic_model="claude-opus-4-6",
        judge_model="gpt-3.5-turbo",
    )
    logger.info("[Setup] Dual-model engine active: GPT-5.4 + Opus 4.6")
else:
    logger.info("[Setup] Single-model mode: gpt-3.5-turbo")

# load the product spec
# TODO: 3 - Load the product spec document Product-Spec-Email-Router.txt into a variable called product_spec
with open(phase_dir / "Product-Spec-Email-Router.txt", "r", encoding="utf-8") as f:
    product_spec = f.read()

# Instantiate all the agents

# Action Planning Agent (complex — uses dual-model when available)
knowledge_action_planning = (
    "When building a development plan from a product specification, always follow this exact order:\n"
    "1. Define user stories from the product specification.\n"
    "2. Group the user stories into product features.\n"
    "3. Create detailed engineering tasks from the user stories and features.\n"
    "Return only these three steps, one per line."
)
# TODO: 4 - Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(
    openai_api_key, knowledge_action_planning, dual_engine=dual_engine
)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several user stories for the product specification below, where the personas are the different users of the product.\n\n"
    # TODO: 5 - Complete this knowledge string by appending the product_spec loaded in TODO 3
    + product_spec
)
# TODO: 6 - Instantiate a product_manager_knowledge_agent
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_product_manager, knowledge_product_manager)

# Product Manager - Evaluation Agent (complex — uses dual-model when available)
# TODO: 7 - Define the persona and evaluation criteria for a Product Manager evaluation agent
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    "You are an evaluation agent that checks the answers of other worker agents",
    'The answer should be user stories that follow the following structure: "As a [type of user], I want [an action or feature] so that [benefit/value]."',
    product_manager_knowledge_agent,
    3,
    dual_engine=dual_engine
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = (
    "Features of a product are defined by organizing similar user stories into cohesive groups. "
    "Return only structured product features grounded in the user stories and product specification that you are given."
)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_program_manager, knowledge_program_manager)

# Program Manager - Evaluation Agent (complex — uses dual-model when available)
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

# TODO: 8 - Instantiate a program_manager_evaluation_agent
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_program_manager_eval,
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user",
    program_manager_knowledge_agent,
    3,
    dual_engine=dual_engine
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = (
    "Development tasks are defined by identifying the concrete engineering work needed to implement each user story and feature. "
    "Return only structured engineering tasks grounded in the user stories, product features, and product specification that you are given."
)
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_dev_engineer, knowledge_dev_engineer)

# Development Engineer - Evaluation Agent (complex — uses dual-model when available)
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
# TODO: 9 - Instantiate a development_engineer_evaluation_agent
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_dev_engineer_eval,
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first",
    development_engineer_knowledge_agent,
    3,
    dual_engine=dual_engine
)


# Job function persona support functions
# TODO: 11 - Define the support functions for the routes of the routing agent
workflow_context = {
    "user_stories": "",
    "features": "",
    "tasks": ""
}


def normalize_section_output(text, heading):
    cleaned_text = text.strip()
    lines = cleaned_text.splitlines()
    if lines and lines[0].strip().lower() == heading.lower():
        return "\n".join(lines[1:]).strip()
    return cleaned_text


def product_manager_support_function(step):
    query = (
        f"{step}\n\n"
        "Create user stories only for the product specification in your knowledge.\n"
        "Return each story on its own line using this exact structure:\n"
        "As a [type of user], I want [an action or feature] so that [benefit/value]."
    )
    response = product_manager_knowledge_agent.respond(query)
    result = product_manager_evaluation_agent.evaluate(query, response)
    if not result.get('success', False):
        logger.warning('Warning: Product Manager evaluation did not pass criteria.')
    final_response = normalize_section_output(result["final_response"], "User Stories:")
    workflow_context["user_stories"] = final_response
    return final_response


def program_manager_support_function(step):
    if not workflow_context["user_stories"]:
        raise RuntimeError("Product features cannot be created before the user stories are available.")

    query = (
        f"{step}\n\n"
        "Use the product specification and user stories below to define product features only.\n\n"
        "Return each feature using this exact structure:\n"
        "Feature Name: [clear, concise title]\n"
        "Description: [what the feature does and its purpose]\n"
        "Key Functionality: [specific capabilities or actions]\n"
        "User Benefit: [how it creates value for the user]\n\n"
        f"Product Specification:\n{product_spec}\n\n"
        f"User Stories:\n{workflow_context['user_stories']}"
    )
    response = program_manager_knowledge_agent.respond(query)
    result = program_manager_evaluation_agent.evaluate(query, response)
    if not result.get('success', False):
        logger.warning('Warning: Program Manager evaluation did not pass criteria.')
    final_response = normalize_section_output(result["final_response"], "Product Features:")
    workflow_context["features"] = final_response
    return final_response


def development_engineer_support_function(step):
    if not workflow_context["user_stories"] or not workflow_context["features"]:
        raise RuntimeError("Engineering tasks require both user stories and product features.")

    query = (
        f"{step}\n\n"
        "Use the product specification, user stories, and product features below to create engineering tasks only.\n\n"
        "Return each task using this exact structure:\n"
        "Task ID: [unique identifier]\n"
        "Task Title: [brief description of work]\n"
        "Related User Story: [reference to parent user story]\n"
        "Description: [detailed technical work required]\n"
        "Acceptance Criteria: [specific completion requirements]\n"
        "Estimated Effort: [time or complexity estimate]\n"
        "Dependencies: [tasks that must be completed first]\n\n"
        f"Product Specification:\n{product_spec}\n\n"
        f"User Stories:\n{workflow_context['user_stories']}\n\n"
        f"Product Features:\n{workflow_context['features']}"
    )
    response = development_engineer_knowledge_agent.respond(query)
    result = development_engineer_evaluation_agent.evaluate(query, response)
    if not result.get('success', False):
        logger.warning('Warning: Development Engineer evaluation did not pass criteria.')
    final_response = normalize_section_output(result["final_response"], "Engineering Tasks:")
    workflow_context["tasks"] = final_response
    return final_response


# Routing Agent
# TODO: 10 - Instantiate a routing_agent
routing_agent = RoutingAgent(openai_api_key, [])
routing_agent.agents = [
    {
        "name": "Product Manager",
        "description": "Product Manager work only: define personas and user stories in the format As a [user], I want [action] so that [benefit]. Do not define features or engineering tasks.",
        "func": lambda x: product_manager_support_function(x)
    },
    {
        "name": "Program Manager",
        "description": "Program Manager work only: group user stories into product features with Feature Name, Description, Key Functionality, and User Benefit. Do not define user stories or engineering tasks.",
        "func": lambda x: program_manager_support_function(x)
    },
    {
        "name": "Development Engineer",
        "description": "Development Engineer work only: create detailed engineering tasks with Task ID, Task Title, Related User Story, Description, Acceptance Criteria, Estimated Effort, and Dependencies. Do not define user stories or product features.",
        "func": lambda x: development_engineer_support_function(x)
    }
]


# Run the workflow

logger.info("\n*** Workflow execution started ***\n")
# Workflow Prompt
workflow_prompt = (
    "Create a complete development plan for the Email Router product, including user stories, "
    "grouped product features, and detailed engineering tasks."
)
default_workflow_steps = [
    "Define user stories from the product specification.",
    "Group the user stories into product features.",
    "Create detailed engineering tasks from the user stories and features."
]
logger.info(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

logger.info("\nDefining workflow steps from the workflow prompt")
# TODO: 12 - Implement the workflow
planned_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
if len(planned_steps) < len(default_workflow_steps):
    planned_steps = default_workflow_steps
workflow_steps = [
    f"Product Manager: {planned_steps[0]}",
    f"Program Manager: {planned_steps[1]}",
    f"Development Engineer: {planned_steps[2]}"
]
logger.info(f"Workflow steps: {workflow_steps}\n")

completed_steps = []

for i, step in enumerate(workflow_steps):
    logger.info(f"\n{'='*60}")
    logger.info(f"Executing Step {i+1}: {step}")
    logger.info(f"{'='*60}")
    result = routing_agent.route(step)
    completed_steps.append({"step": step, "result": result})
    logger.info(f"\nStep {i+1} Result:\n{result}")

logger.info(f"\n{'='*60}")
logger.info("*** Workflow execution completed ***")
logger.info(f"{'='*60}")
logger.info("\nFinal Development Plan:")
logger.info(f"\nUser Stories:\n{workflow_context['user_stories']}")
logger.info(f"\nProduct Features:\n{workflow_context['features']}")
logger.info(f"\nEngineering Tasks:\n{workflow_context['tasks']}")
