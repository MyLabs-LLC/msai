import re

def apply_patch(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Imports
    content = content.replace(
        "import ast\nimport numpy as np",
        "import ast\nimport logging\nimport numpy as np"
    )
    content = content.replace(
        "from datetime import datetime\n",
        "from datetime import datetime\nfrom typing import List, Dict, Any, Optional\nfrom tenacity import retry, stop_after_attempt, wait_exponential\n\nlogger = logging.getLogger(__name__)\n"
    )

    # DirectPromptAgent
    content = content.replace(
        "def __init__(self, openai_api_key):",
        "def __init__(self, openai_api_key: str, model: str = \"gpt-3.5-turbo\"):"
    )
    content = content.replace(
        "self.openai_api_key = openai_api_key\n        self.client = OpenAI(api_key=self.openai_api_key)",
        "self.openai_api_key = openai_api_key\n        self.model = model\n        self.client = OpenAI(api_key=self.openai_api_key)"
    )
    content = content.replace(
        "def respond(self, prompt):",
        "@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))\n    def respond(self, prompt: str) -> str:"
    )
    content = content.replace(
        "model=\"gpt-3.5-turbo\",",
        "model=self.model,"
    )

    # AugmentedPromptAgent
    content = content.replace(
        "def __init__(self, openai_api_key, persona):",
        "def __init__(self, openai_api_key: str, persona: str, model: str = \"gpt-3.5-turbo\"):"
    )
    content = content.replace(
        "self.openai_api_key = openai_api_key\n        self.client = OpenAI(api_key=self.openai_api_key)",
        "self.openai_api_key = openai_api_key\n        self.model = model\n        self.client = OpenAI(api_key=self.openai_api_key)"
    )
    content = content.replace(
        "def respond(self, input_text):",
        "@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))\n    def respond(self, input_text: str) -> str:"
    )

    # KnowledgeAugmentedPromptAgent
    content = content.replace(
        "def __init__(self, openai_api_key, persona, knowledge):",
        "def __init__(self, openai_api_key: str, persona: str, knowledge: str, model: str = \"gpt-3.5-turbo\"):"
    )
    content = content.replace(
        "self.openai_api_key = openai_api_key\n        self.client = OpenAI(api_key=self.openai_api_key)",
        "self.openai_api_key = openai_api_key\n        self.model = model\n        self.client = OpenAI(api_key=self.openai_api_key)"
    )

    # RAGKnowledgePromptAgent
    content = content.replace(
        "def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):",
        "def __init__(self, openai_api_key: str, persona: str, chunk_size: int = 2000, chunk_overlap: int = 100, model: str = \"gpt-3.5-turbo\", embedding_model: str = \"text-embedding-3-large\"):"
    )
    content = content.replace(
        "self.openai_api_key = openai_api_key\n        self.unique_filename = f\"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv\"\n        self.client = OpenAI(api_key=self.openai_api_key)",
        "self.openai_api_key = openai_api_key\n        self.model = model\n        self.embedding_model = embedding_model\n        self.client = OpenAI(api_key=self.openai_api_key)\n        self.df: Optional[pd.DataFrame] = None"
    )
    content = content.replace(
        "def get_embedding(self, text):",
        "@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))\n    def get_embedding(self, text: str) -> List[float]:"
    )
    content = content.replace(
        "model=\"text-embedding-3-large\",",
        "model=self.embedding_model,"
    )
    content = content.replace(
        "def calculate_similarity(self, vector_one, vector_two):",
        "def calculate_similarity(self, vector_one: List[float], vector_two: List[float]) -> float:"
    )
    content = content.replace(
        "return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))",
        "return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))"
    )
    content = content.replace(
        "def chunk_text(self, text):",
        "def chunk_text(self, text: str) -> List[Dict[str, Any]]:"
    )
    content = content.replace(
        "with open(f\"chunks-{self.unique_filename}\", 'w', newline='', encoding='utf-8') as csvfile:\n            writer = csv.DictWriter(csvfile, fieldnames=[\"text\", \"chunk_size\"])\n            writer.writeheader()\n            for chunk in chunks:\n                writer.writerow({k: chunk[k] for k in [\"text\", \"chunk_size\"]})",
        "self.df = pd.DataFrame(chunks)"
    )
    content = content.replace(
        "def calculate_embeddings(self):",
        "def calculate_embeddings(self) -> pd.DataFrame:"
    )
    content = content.replace(
        "df = pd.read_csv(f\"chunks-{self.unique_filename}\", encoding='utf-8')\n        df['embeddings'] = df['text'].apply(self.get_embedding)\n        df.to_csv(f\"embeddings-{self.unique_filename}\", encoding='utf-8', index=False)\n        return df",
        "if self.df is None:\n            raise ValueError(\"Chunks not initialized. Call chunk_text first.\")\n        self.df['embeddings'] = self.df['text'].apply(self.get_embedding)\n        return self.df"
    )
    content = content.replace(
        "def find_prompt_in_knowledge(self, prompt):",
        "@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))\n    def find_prompt_in_knowledge(self, prompt: str) -> str:"
    )
    content = content.replace(
        "prompt_embedding = self.get_embedding(prompt)\n        df = pd.read_csv(f\"embeddings-{self.unique_filename}\", encoding='utf-8')\n        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(ast.literal_eval(x)))\n        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))\n\n        best_chunk = df.loc[df['similarity'].idxmax(), 'text']",
        "if self.df is None or 'embeddings' not in self.df.columns:\n            raise ValueError(\"Embeddings not initialized. Call calculate_embeddings first.\")\n            \n        prompt_embedding = self.get_embedding(prompt)\n        self.df['similarity'] = self.df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))\n\n        best_chunk = self.df.loc[self.df['similarity'].idxmax(), 'text']"
    )

    # EvaluationAgent
    content = content.replace(
        "def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions):",
        "def __init__(self, openai_api_key: str, persona: str, evaluation_criteria: str, worker_agent: Any, max_interactions: int, model: str = \"gpt-3.5-turbo\"):"
    )
    content = content.replace(
        "self.max_interactions = max_interactions\n        self.client = OpenAI(api_key=self.openai_api_key)",
        "self.max_interactions = max_interactions\n        self.model = model\n        self.client = OpenAI(api_key=self.openai_api_key)"
    )
    content = content.replace(
        "def evaluate(self, initial_prompt, candidate_response=None):",
        "@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))\n    def evaluate(self, initial_prompt: str, candidate_response: Optional[str] = None) -> Dict[str, Any]:"
    )
    content = content.replace(
        "evaluation = \"\"\n\n        for i in range(self.max_interactions):",
        "evaluation = \"\"\n        success = False\n\n        for i in range(self.max_interactions):"
    )
    content = content.replace("print(f\"\\n--- Interaction {i+1} ---\")", "logger.info(f\"--- Interaction {i+1} ---\")")
    content = content.replace("print(\" Step 1: Worker agent generates a response to the prompt\")", "logger.info(\"Step 1: Worker agent generates a response to the prompt\")")
    content = content.replace("print(f\"Prompt:\\n{prompt_to_evaluate}\")", "logger.debug(f\"Prompt:\\n{prompt_to_evaluate}\")")
    content = content.replace("print(\" Step 1: Evaluating the provided worker response\")", "logger.info(\"Step 1: Evaluating the provided worker response\")")
    content = content.replace("print(f\"Prompt:\\n{initial_prompt}\")", "logger.debug(f\"Prompt:\\n{initial_prompt}\")")
    content = content.replace("print(f\"Worker Agent Response:\\n{response_from_worker}\")", "logger.debug(f\"Worker Agent Response:\\n{response_from_worker}\")")
    content = content.replace("print(\" Step 2: Evaluator agent judges the response\")", "logger.info(\"Step 2: Evaluator agent judges the response\")")
    content = content.replace("print(f\"Evaluator Agent Evaluation:\\n{evaluation}\")", "logger.debug(f\"Evaluator Agent Evaluation:\\n{evaluation}\")")
    content = content.replace("print(\" Step 3: Check if evaluation is positive\")", "logger.info(\"Step 3: Check if evaluation is positive\")")
    content = content.replace("print(\"Final solution accepted.\")", "logger.info(\"Final solution accepted.\")\n                success = True")
    content = content.replace("print(\" Step 4: Generate instructions to correct the response\")", "logger.info(\"Step 4: Generate instructions to correct the response\")")
    content = content.replace("print(f\"Instructions to fix:\\n{instructions}\")", "logger.debug(f\"Instructions to fix:\\n{instructions}\")")
    content = content.replace("print(\" Step 5: Send feedback to worker agent for refinement\")", "logger.info(\"Step 5: Send feedback to worker agent for refinement\")")
    
    content = content.replace(
        "\"iterations\": i + 1\n        }",
        "\"iterations\": i + 1,\n            \"success\": success\n        }"
    )

    # RoutingAgent
    content = content.replace(
        "def __init__(self, openai_api_key, agents=None):",
        "def __init__(self, openai_api_key: str, agents: Optional[List[Dict[str, Any]]] = None, embedding_model: str = \"text-embedding-3-large\"):"
    )
    content = content.replace(
        "self.agents = agents or []\n        self.client = OpenAI(api_key=self.openai_api_key)",
        "self.agents = agents or []\n        self.embedding_model = embedding_model\n        self.client = OpenAI(api_key=self.openai_api_key)"
    )
    content = content.replace(
        "def route(self, user_input):",
        "def route(self, user_input: str) -> Any:"
    )
    content = content.replace("print(f\"[Router] Explicit role match: {agent['name']}\")", "logger.info(f\"[Router] Explicit role match: {agent['name']}\")")
    content = content.replace("print(similarity)", "logger.debug(f\"Similarity for {agent['name']}: {similarity}\")")
    content = content.replace("print(f\"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})\")", "logger.info(f\"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})\")")

    # ActionPlanningAgent
    content = content.replace(
        "def __init__(self, openai_api_key, knowledge):",
        "def __init__(self, openai_api_key: str, knowledge: str, model: str = \"gpt-3.5-turbo\"):"
    )
    content = content.replace(
        "self.knowledge = knowledge\n        self.client = OpenAI(api_key=self.openai_api_key)",
        "self.knowledge = knowledge\n        self.model = model\n        self.client = OpenAI(api_key=self.openai_api_key)"
    )
    content = content.replace(
        "def extract_steps_from_prompt(self, prompt):",
        "@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))\n    def extract_steps_from_prompt(self, prompt: str) -> List[str]:"
    )

    with open(file_path, 'w') as f:
        f.write(content)

apply_patch('ProjectManagement/starter/phase_1/workflow_agents/base_agents.py')
apply_patch('ProjectManagement/starter/phase_2/workflow_agents/base_agents.py')
