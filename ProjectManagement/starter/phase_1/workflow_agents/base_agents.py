from openai import OpenAI
import anthropic as anthropic_sdk
import logging
import numpy as np
import pandas as pd
import re
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class DualModelQueryEngine:
    """
    Queries both an OpenAI model (GPT-5.4) and an Anthropic model (Opus 4.6) with
    identical prompts, logs the full output from each, and selects the best response
    via a lightweight judge call. Used for complex agent decisions that involve 3+
    reasoning steps (evaluation loops, action planning, correction generation).
    """

    def __init__(
        self,
        openai_api_key: str,
        anthropic_api_key: str,
        openai_model: str = "gpt-5.4",
        anthropic_model: str = "claude-opus-4-6",
        judge_model: str = "gpt-3.5-turbo",
    ):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.anthropic_client = anthropic_sdk.Anthropic(api_key=anthropic_api_key)
        self.openai_model = openai_model
        self.anthropic_model = anthropic_model
        self.judge_model = judge_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _query_openai(self, system: str, user: str, temperature: float = 0) -> str:
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _query_anthropic(self, system: str, user: str, temperature: float = 0) -> str:
        response = self.anthropic_client.messages.create(
            model=self.anthropic_model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature
        )
        return response.content[0].text.strip()

    def _select_best(self, openai_resp: str, anthropic_resp: str, criteria: str) -> Dict[str, str]:
        """Use a lightweight judge model to pick the stronger response."""
        judge_prompt = (
            "You are a response quality judge. Compare the two candidate responses below "
            "and decide which one better satisfies the given criteria.\n"
            "Respond with EXACTLY one line: 'SELECTED: A' or 'SELECTED: B' "
            "followed by a brief reason on the next line starting with 'REASON: '.\n\n"
            f"Criteria: {criteria}\n\n"
            f"--- Response A (OpenAI {self.openai_model}) ---\n{openai_resp}\n\n"
            f"--- Response B (Anthropic {self.anthropic_model}) ---\n{anthropic_resp}"
        )
        judge_resp = self.openai_client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0
        )
        verdict = judge_resp.choices[0].message.content.strip()

        if "SELECTED: B" in verdict.upper():
            selected = anthropic_resp
            selected_label = f"Anthropic ({self.anthropic_model})"
        else:
            selected = openai_resp
            selected_label = f"OpenAI ({self.openai_model})"

        reason_line = ""
        for line in verdict.splitlines():
            if line.strip().upper().startswith("REASON:"):
                reason_line = line.strip()[len("REASON:"):].strip()
                break
        if not reason_line:
            reason_line = verdict

        return {"selected": selected, "selected_label": selected_label, "reason": reason_line}

    def query(self, system: str, user: str, temperature: float = 0, criteria: str = "") -> Dict[str, Any]:
        """
        Query both models with identical prompts, log full outputs from each,
        and return the best response selected by a judge.
        """
        logger.info(f"\n{'─'*70}")
        logger.info("[DualModel] Querying both models for consensus...")
        logger.info(f"[DualModel]   OpenAI model   : {self.openai_model}")
        logger.info(f"[DualModel]   Anthropic model : {self.anthropic_model}")
        logger.info(f"{'─'*70}")

        openai_resp = self._query_openai(system, user, temperature)
        logger.info(f"\n[DualModel] ═══ OpenAI ({self.openai_model}) — Full Output ═══")
        logger.info(openai_resp)

        anthropic_resp = self._query_anthropic(system, user, temperature)
        logger.info(f"\n[DualModel] ═══ Anthropic ({self.anthropic_model}) — Full Output ═══")
        logger.info(anthropic_resp)

        if not criteria:
            criteria = "Overall quality, completeness, structure, and adherence to the task."

        logger.info(f"\n[DualModel] ── Judge ({self.judge_model}) evaluating both responses ──")
        result = self._select_best(openai_resp, anthropic_resp, criteria)
        logger.info(f"[DualModel] ✓ Winner : {result['selected_label']}")
        logger.info(f"[DualModel]   Reason : {result['reason']}")
        logger.info(f"{'─'*70}\n")

        return {
            "openai_response": openai_resp,
            "anthropic_response": anthropic_resp,
            "selected": result["selected"],
            "selected_label": result["selected_label"],
            "reason": result["reason"],
        }


# DirectPromptAgent class definition
class DirectPromptAgent:

    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo"):
        self.openai_api_key = openai_api_key
        self.model = model
        self.client = OpenAI(api_key=self.openai_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def respond(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content


# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    def __init__(self, openai_api_key: str, persona: str, model: str = "gpt-3.5-turbo"):
        """Initialize the agent with given attributes."""
        self.persona = persona
        self.openai_api_key = openai_api_key
        self.model = model
        self.client = OpenAI(api_key=self.openai_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def respond(self, input_text: str) -> str:
        """Generate a response using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are {self.persona}. Forget all previous conversational context."},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )

        return response.choices[0].message.content


# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key: str, persona: str, knowledge: str, model: str = "gpt-3.5-turbo"):
        """Initialize the agent with provided attributes."""
        self.persona = persona
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key
        self.model = model
        self.client = OpenAI(api_key=self.openai_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def respond(self, input_text: str) -> str:
        """Generate a response using the OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    f"You are {self.persona} knowledge-based assistant. Forget all previous context. "
                    f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge} "
                    f"Answer the prompt based on this knowledge, not your own."
                )},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key: str, persona: str, chunk_size: int = 2000, chunk_overlap: int = 100, model: str = "gpt-3.5-turbo", embedding_model: str = "text-embedding-3-large"):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.model = model
        self.embedding_model = embedding_model
        self.client = OpenAI(api_key=self.openai_api_key)
        self.df: Optional[pd.DataFrame] = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_embedding(self, text: str) -> List[float]:
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one: List[float], vector_two: List[float]) -> float:
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\r\n?', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text) and separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            if end <= start:
                end = min(start + self.chunk_size, len(text))

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            if end >= len(text):
                break

            start = max(end - self.chunk_overlap, start + 1)
            chunk_id += 1

        self.df = pd.DataFrame(chunks)

        return chunks

    def calculate_embeddings(self) -> pd.DataFrame:
        """
        Calculates embeddings for each chunk and stores them in-memory.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        if self.df is None:
            raise ValueError("Chunks not initialized. Call chunk_text first.")
        self.df['embeddings'] = self.df['text'].apply(self.get_embedding)
        return self.df

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def find_prompt_in_knowledge(self, prompt: str) -> str:
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        if self.df is None or 'embeddings' not in self.df.columns:
            raise ValueError("Embeddings not initialized. Call calculate_embeddings first.")
            
        prompt_embedding = self.get_embedding(prompt)
        self.df['similarity'] = self.df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = self.df.loc[self.df['similarity'].idxmax(), 'text']

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content


# EvaluationAgent class definition
class EvaluationAgent:
    """
    Quality-gate agent that evaluates worker responses against criteria.
    Complex agent (3+ decisions per cycle: evaluate, judge pass/fail, generate corrections,
    feed back to worker). When a DualModelQueryEngine is provided, all evaluation and
    correction LLM calls are sent to both GPT-5.4 and Opus 4.6, with the best response
    selected and both outputs fully logged.
    """

    def __init__(self, openai_api_key: str, persona: str, evaluation_criteria: str,
                 agent_to_evaluate: Any, max_interactions: int, model: str = "gpt-3.5-turbo",
                 dual_engine: Optional[DualModelQueryEngine] = None):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.agent_to_evaluate = agent_to_evaluate
        self.max_interactions = max_interactions
        self.model = model
        self.dual_engine = dual_engine
        self.client = OpenAI(api_key=self.openai_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm_single(self, system: str, user: str) -> str:
        """Single-model fallback for when no dual engine is configured."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    def _call_llm(self, system: str, user: str, criteria_hint: str = "") -> str:
        """
        Route LLM calls through the dual-model engine when available.
        Falls back to single-model (gpt-3.5-turbo) otherwise.
        """
        if self.dual_engine:
            result = self.dual_engine.query(
                system, user, temperature=0,
                criteria=criteria_hint or self.evaluation_criteria
            )
            return result["selected"]
        return self._call_llm_single(system, user)

    def evaluate(self, initial_prompt: str, candidate_response: Optional[str] = None) -> Dict[str, Any]:
        prompt_to_evaluate = initial_prompt
        response_from_worker = candidate_response
        evaluation = ""
        success = False

        for i in range(self.max_interactions):
            logger.info(f"--- Interaction {i+1} ---")

            if response_from_worker is None:
                logger.info("Step 1: Worker agent generates a response to the prompt")
                logger.debug(f"Prompt:\n{prompt_to_evaluate}")
                response_from_worker = self.agent_to_evaluate.respond(prompt_to_evaluate)
            else:
                logger.info("Step 1: Evaluating the provided worker response")
                logger.debug(f"Prompt:\n{initial_prompt}")
            logger.debug(f"Worker Agent Response:\n{response_from_worker}")

            logger.info("Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            evaluation = self._call_llm(
                self.persona, eval_prompt,
                criteria_hint="Accurate pass/fail judgment of whether the response meets the stated criteria."
            )
            logger.debug(f"Evaluator Agent Evaluation:\n{evaluation}")

            logger.info("Step 3: Check if evaluation is positive")
            evaluation_text = evaluation.lower()
            if evaluation_text.startswith("yes") or evaluation_text.startswith("pass") or "meets the criteria" in evaluation_text:
                logger.info("Final solution accepted.")
                success = True
                break
            else:
                logger.info("Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                instructions = self._call_llm(
                    self.persona, instruction_prompt,
                    criteria_hint="Clear, actionable correction instructions that address all identified issues."
                )
                logger.debug(f"Instructions to fix:\n{instructions}")

                logger.info("Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
                response_from_worker = None
        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iterations": i + 1,
            "success": success
        }


# RoutingAgent class definition
class RoutingAgent():

    def __init__(self, openai_api_key: str, agents: Optional[List[Dict[str, Any]]] = None, embedding_model: str = "text-embedding-3-large"):
        self.openai_api_key = openai_api_key
        self.agents = agents or []
        self.embedding_model = embedding_model
        self.client = OpenAI(api_key=self.openai_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        embedding = response.data[0].embedding
        return embedding

    def cosine_similarity(self, vector_one, vector_two):
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def route(self, user_input: str) -> Any:
        normalized_input = user_input.lower()
        for agent in self.agents:
            if agent["name"].lower() in normalized_input:
                logger.info(f"[Router] Explicit role match: {agent['name']}")
                return agent["func"](user_input)

        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = agent.get("_description_embedding")
            if agent_emb is None:
                agent_emb = self.get_embedding(agent["description"])
                agent["_description_embedding"] = agent_emb

            similarity = self.cosine_similarity(input_emb, agent_emb)
            logger.debug(f"Similarity for {agent['name']}: {similarity}")

            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        logger.info(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


# ActionPlanningAgent class definition
class ActionPlanningAgent:
    """
    Extracts actionable steps from a high-level prompt using domain knowledge.
    Complex agent (3+ decisions: interpret prompt, identify steps, order them,
    format output). When a DualModelQueryEngine is provided, both GPT-5.4 and
    Opus 4.6 generate the step list, and the best extraction is selected.
    """

    def __init__(self, openai_api_key: str, knowledge: str, model: str = "gpt-3.5-turbo",
                 dual_engine: Optional[DualModelQueryEngine] = None):
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge
        self.model = model
        self.dual_engine = dual_engine
        self.client = OpenAI(api_key=self.openai_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm_single(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        )
        return response.choices[0].message.content

    def extract_steps_from_prompt(self, prompt: str) -> List[str]:
        system_msg = (
            "You are an action planning agent. Using your knowledge, you extract from the user prompt "
            "the steps requested to complete the action the user is asking for. You return the steps as a list. "
            "Only return the steps in your knowledge. Forget any previous context. "
            f"This is your knowledge: {self.knowledge}"
        )

        if self.dual_engine:
            result = self.dual_engine.query(
                system_msg, prompt,
                criteria="Clear, correctly ordered, actionable steps that faithfully reflect the provided knowledge."
            )
            response_text = result["selected"]
        else:
            response_text = self._call_llm_single(system_msg, prompt)

        raw_steps = [line.strip() for line in response_text.splitlines() if line.strip()]
        steps = [re.sub(r"^\s*(?:(?i:step)\s+\d+[\).\:-]?|\d+[\).\:-]?|[-*])\s*", "", line).strip() for line in raw_steps]

        return steps
