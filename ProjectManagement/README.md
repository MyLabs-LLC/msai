# AI-Powered Agentic Workflow for Project Management

Welcome to the **AI-Powered Agentic Workflow for Project Management** repository. This project demonstrates how to build a robust, reusable library of Large Language Model (LLM) agents and orchestrate them into a complex, multi-step workflow.

The pilot use case for this system is automating the technical project management pipeline for **InnovateNext Solutions**, specifically transforming a raw product specification (the "Email Router") into structured User Stories, Product Features, and Engineering Tasks.

---

## Project Overview

This project is divided into two distinct phases:

1. **Phase 1: The Agentic Toolkit**
   A reusable Python library (`workflow_agents/base_agents.py`) containing seven distinct AI agent classes. Each agent is designed with a specific interaction pattern (e.g., direct prompting, persona adoption, knowledge augmentation, retrieval-augmented generation, evaluation, routing, and action planning).

2. **Phase 2: The Orchestrated Workflow**
   A production-ready script (`agentic_workflow.py`) that chains these agents together to simulate a real-world product development pipeline. It uses an Action Planning Agent to define steps, a Routing Agent to delegate tasks, and specialized Knowledge/Evaluation Agent pairs to act as a Product Manager, Program Manager, and Development Engineer.

---

## Directory Structure

```
ProjectManagement/
├── README.md
├── requirements.txt
├── project_overview.md
├── reflection.md
├── .gitignore
├── starter/
│   ├── phase_1/
│   │   ├── workflow_agents/
│   │   │   ├── __init__.py                          # Empty, makes it a package
│   │   │   └── base_agents.py                       # ALL 7 agent classes
│   │   ├── direct_prompt_agent.py                   # Test script
│   │   ├── augmented_prompt_agent.py                # Test script
│   │   ├── knowledge_augmented_prompt_agent.py      # Test script
│   │   ├── rag_knowledge_prompt_agent.py            # Test script (provided)
│   │   ├── evaluation_agent.py                      # Test script
│   │   ├── routing_agent.py                         # Test script
│   │   ├── action_planning_agent.py                 # Test script
│   │   └── *_output.txt                             # Terminal output for each test
│   └── phase_2/
│       ├── workflow_agents/
│       │   ├── __init__.py
│       │   └── base_agents.py                       # Copy from phase_1
│       ├── agentic_workflow.py                      # Main workflow script
│       ├── Product-Spec-Email-Router.txt            # Input product spec
│       └── workflow_output.txt                      # Terminal output of full run
```

---

## Architecture & Agent Library

### The 7 Core Agents (`base_agents.py`)

| # | Agent Class | Purpose | Primary Method | Returns |
|---|---|---|---|---|
| 1 | `DirectPromptAgent` | Relays a prompt directly to the LLM without any system prompt or persona. | `respond(prompt)` | `str` |
| 2 | `AugmentedPromptAgent` | Adopts a specific persona via a system prompt before answering. Instructs the LLM to forget previous conversational context. | `respond(input_text)` | `str` |
| 3 | `KnowledgeAugmentedPromptAgent` | Adopts a persona *and* is injected with specific knowledge. Instructed to answer solely based on that provided knowledge, not the LLM's training data. | `respond(input_text)` | `str` |
| 4 | `RAGKnowledgePromptAgent` | Implements Retrieval-Augmented Generation. Chunks large texts, calculates embeddings, and uses cosine similarity to find the most relevant context to answer a prompt. *(Provided — not student-implemented.)* | `find_prompt_in_knowledge(prompt)` | `str` |
| 5 | `EvaluationAgent` | Acts as a quality gate. Reviews the output of a "worker" agent against strict criteria. If the output fails, generates correction instructions and forces the worker to refine the answer (up to `max_interactions`). Uses `temperature=0` for all evaluation and correction calls. | `evaluate(initial_prompt)` | `dict` (`final_response`, `evaluation`, `iterations`, `success`) |
| 6 | `RoutingAgent` | Dynamically routes a prompt to the most capable agent in a pool. Uses `text-embedding-3-large` embeddings and cosine similarity to match the prompt against each agent's description. Returns the selected agent's response. | `route(user_input)` | varies |
| 7 | `ActionPlanningAgent` | Analyzes a high-level goal and extracts a sequential list of actionable steps. System prompt defines it as an "Action Planning Agent" and injects domain knowledge. | `extract_steps_from_prompt(prompt)` | `List[str]` |

### LLM Configuration

- **Chat Model:** `gpt-3.5-turbo` (default for all agents; configurable via `model` parameter)
- **Embedding Model:** `text-embedding-3-large` (used by `RoutingAgent` and `RAGKnowledgePromptAgent`)
- **API Key:** Passed at initialization — never hardcoded inside agent classes
- **Temperature:** `0` for `EvaluationAgent` evaluation and correction calls; default for other agents

### Dual-Model Consensus (Complex Agents)

Agents that make **3+ decisions per query** — `EvaluationAgent` and `ActionPlanningAgent` — can optionally use a **dual-model consensus** engine powered by both **OpenAI GPT-5.4** and **Anthropic Opus 4.6**:

1. The same system/user prompt is sent to both models simultaneously.
2. Both full responses are logged at INFO level so you can inspect each model's reasoning.
3. A lightweight judge (`gpt-3.5-turbo`) compares the two outputs against task-specific criteria and selects the stronger response.
4. The winner, the losing response, and the judge's reasoning are all logged.

This is implemented via the `DualModelQueryEngine` class in `base_agents.py`. Simple agents (`DirectPromptAgent`, `AugmentedPromptAgent`, `KnowledgeAugmentedPromptAgent`) remain single-model since they make only 1–2 decisions per call.

To enable dual-model mode, add `ANTHROPIC_API_KEY` to your `.env` file. If the key is missing, the system automatically falls back to single-model (`gpt-3.5-turbo`) with a warning.

---

## Phase 2 Workflow: How It Works

The `agentic_workflow.py` script orchestrates these agents into a PM → PgM → Dev pipeline:

1. **Action Planning:** The `ActionPlanningAgent` breaks a high-level prompt into 3 ordered steps: define user stories, group into features, create engineering tasks.
2. **Routing:** Each step is sent to the `RoutingAgent`, which matches it to the correct specialist role (Product Manager, Program Manager, or Development Engineer).
3. **Generate + Evaluate (Dual-Model):** Each specialist consists of a `KnowledgeAugmentedPromptAgent` (to generate) paired with an `EvaluationAgent` (to validate). The evaluation and correction calls are sent to **both GPT-5.4 and Opus 4.6**, with the best judgment selected. If the output doesn't meet criteria, correction instructions are generated (also dual-model) and the worker refines its answer — up to 3 iterations.
4. **Contextual Pipeline:** User stories from the PM are injected into the PgM prompt. Both user stories and features are injected into the Dev prompt. This ensures downstream outputs are grounded in upstream decisions.
5. **Final Output:** A structured development plan containing User Stories, Product Features, and Engineering Tasks.

### Support Functions

Three support functions wire the generate-evaluate pattern:

```python
def product_manager_support_function(query):
    response = product_manager_knowledge_agent.respond(query)
    result = product_manager_evaluation_agent.evaluate(query, response)
    return result["final_response"]
```

Each accepts a step from the action plan, calls `respond()` on the knowledge agent, passes the response to `evaluate()` on the evaluation agent, and returns the validated `final_response`.

### Routing Agent Configuration

The routing agent is configured with three routes, each containing a `name`, `description`, and `func`:

- **Product Manager:** Defines personas and user stories in the format `As a [user], I want [action] so that [benefit]`.
- **Program Manager:** Groups user stories into features with `Feature Name`, `Description`, `Key Functionality`, and `User Benefit`.
- **Development Engineer:** Creates engineering tasks with `Task ID`, `Task Title`, `Related User Story`, `Description`, `Acceptance Criteria`, `Estimated Effort`, and `Dependencies`.

---

## Expected Output Structures

### User Stories (Product Manager)
```
As a [type of user], I want [an action or feature] so that [benefit/value].
```

### Product Features (Program Manager)
```
Feature Name: [clear, concise title]
Description: [what the feature does and its purpose]
Key Functionality: [specific capabilities or actions]
User Benefit: [how it creates value for the user]
```

### Engineering Tasks (Development Engineer)
```
Task ID: [unique identifier]
Task Title: [brief description of work]
Related User Story: [reference to parent user story]
Description: [detailed technical work required]
Acceptance Criteria: [specific completion requirements]
Estimated Effort: [time or complexity estimate]
Dependencies: [tasks that must be completed first]
```

---

## Engineering Improvements Beyond Starter Code

This codebase has been refactored from the initial starter to meet professional standards:

- **API Resilience:** All OpenAI API calls are wrapped with `tenacity` for exponential backoff and retry logic (up to 3 attempts), gracefully handling rate limits and transient network timeouts.
- **In-Memory RAG Processing:** The `RAGKnowledgePromptAgent` processes chunks and embeddings entirely in-memory via `pandas.DataFrame`, eliminating disk I/O and preventing orphaned `.csv` files.
- **Professional Logging:** `print()` statements replaced with Python's `logging` module for configurable verbosity (INFO vs. DEBUG).
- **Configurable Models:** All agents accept a `model` parameter (defaulting to `gpt-3.5-turbo`), making it trivial to swap to `gpt-4o` or `o3-mini` without modifying the library.
- **Type Hints:** Comprehensive Python type hints (`List`, `Dict`, `Optional`, `Any`) throughout the library.
- **Evaluation Success Flag:** `EvaluationAgent` returns a `success` boolean so the workflow can detect when max iterations were exhausted without passing criteria.
- **Embedding Caching:** `RoutingAgent` caches description embeddings on first use, avoiding redundant API calls on repeated routing.
- **Dual-Model Consensus:** Complex agents (`EvaluationAgent`, `ActionPlanningAgent`) query both GPT-5.4 and Opus 4.6, log both full outputs, and a judge selects the best response — reducing single-model bias and improving decision quality.

---

## Setup & Installation

### 1. Prerequisites
- Python 3.8+
- An active [OpenAI API Key](https://platform.openai.com/)
- An [Anthropic API Key](https://console.anthropic.com/) (optional — enables dual-model consensus)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `openai`, `pandas`, `numpy`, `python-dotenv`, `tenacity`, `anthropic`

### 3. Environment Configuration

Create a `.env` file in the root of the workspace (or inside the `ProjectManagement` folder). The scripts intelligently search upward from their location to find it.

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-actual-api-key-here
```

> If `ANTHROPIC_API_KEY` is omitted, the system runs in single-model mode with a warning.

---

## Usage & Execution

### Phase 1: Test Each Agent Individually

```bash
cd starter/phase_1
python direct_prompt_agent.py
python augmented_prompt_agent.py
python knowledge_augmented_prompt_agent.py
python rag_knowledge_prompt_agent.py
python evaluation_agent.py
python routing_agent.py
python action_planning_agent.py
```

### Phase 2: Run the Full Agentic Workflow

```bash
cd starter/phase_2
python agentic_workflow.py
```

### Capturing Output for Submission

To save terminal output to a text file while also displaying it:

```bash
python agentic_workflow.py 2>&1 | tee workflow_output.txt
```

---

## Grading Rubric Quick-Reference

| Category | Must Have |
|----------|----------|
| Agent classes | All 6 implemented in `base_agents.py` with correct `__init__` and primary methods |
| LLM config | `gpt-3.5-turbo` for chat, `text-embedding-3-large` for embeddings, `temperature=0` for EvaluationAgent |
| API key | Passed at init, never hardcoded |
| Test scripts | 7 scripts (one per agent incl. RAG), each imports from `workflow_agents.base_agents`, instantiates, calls, prints |
| Test evidence | Screenshots or text files of terminal output for all 7 |
| Workflow setup | Imports, .env loading, product spec loading all correct |
| Agent instantiation | All knowledge agents, eval agents, routing agent with correct params |
| Support functions | 3 functions, each chains knowledge agent → eval agent → return final_response |
| Workflow logic | Action planning → iterate steps → route each → collect results → print final |
| Output quality | User stories, features, tasks all match required structures |
| Code quality | Descriptive names, snake_case/PascalCase, comments/docstrings, modular |

---

## License

This project is provided for educational and demonstration purposes. Please refer to the `LICENSE.md` file (if applicable) for distribution and modification rights.
