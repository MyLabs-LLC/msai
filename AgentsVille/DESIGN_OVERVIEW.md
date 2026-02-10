# AgentsVille Trip Planner — Design Overview

## Key Decision: Use Anthropic Claude Instead of OpenAI

All OpenAI SDK usage has been replaced with the **Anthropic Python SDK**. The default model is `claude-opus-4-6`. The API key is loaded from a `.env` file.

**SDK integration pattern:**

The project keeps `project_lib.py` unchanged. Instead, `do_chat_completion` is monkey-patched at runtime so that the existing `ChatAgent` class and all helper functions transparently call the Anthropic API:

```python
import anthropic
import project_lib
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=API_KEY)

def do_chat_completion_anthropic(messages, model=None, client=None, **kwargs):
    # Separates system messages into the 'system' parameter (Anthropic convention)
    # Passes remaining messages as conversation history
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]
    response = client.messages.create(
        model=str(model),
        max_tokens=8192,
        system="\n\n".join(system_parts),
        messages=conversation,
    )
    return response.content[0].text

project_lib.do_chat_completion = do_chat_completion_anthropic
```

**Install:** `pip install anthropic` (in addition to `openai`, which is still needed for the bonus narrative/TTS feature)

---

## Model Configuration

Three Anthropic models are available via the `AnthropicModel` enum:

| Enum Value      | Model ID                       | Usage                                      |
|-----------------|--------------------------------|---------------------------------------------|
| `CLAUDE_OPUS`   | `claude-opus-4-6`              | Default — itinerary generation & revision   |
| `CLAUDE_SONNET` | `claude-sonnet-4-5-20250929`   | Balanced speed/intelligence alternative     |
| `CLAUDE_HAIKU`  | `claude-haiku-4-5-20251001`    | High-frequency, low-cost tasks (weather eval) |

- **Default model (`MODEL`)**: `CLAUDE_OPUS`
- **Weather-activity compatibility eval**: `CLAUDE_HAIKU` (fast and cheap for many per-activity calls)
- **Traveler feedback eval**: `CLAUDE_OPUS` (needs strong reasoning)

---

## Project Structure

```
AgentsVille/
├── project_starter.ipynb   # Main notebook — all implementation goes here
├── project_lib.py          # Provided: ChatAgent, mock APIs, utilities (DO NOT MODIFY)
├── .env                    # ANTHROPIC_API_KEY stored here
└── DESIGN_OVERVIEW.md      # This file
```

---

## Architecture: 8 Phases

### Phase 1: Setup & Configuration
- Load `ANTHROPIC_API_KEY` from `.env` using `python-dotenv`
- Initialize `anthropic.Anthropic` client
- Monkey-patch `project_lib.do_chat_completion` to route through the Anthropic SDK
- Define `AnthropicModel` enum with three model tiers

### Phase 2: Define Vacation Details (Pydantic)
- `Traveler` model: `name: str`, `age: int`, `interests: List[Interest]`
- `VacationInfo` model: `travelers: List[Traveler]`, `destination: str`, `date_of_arrival: datetime.date`, `date_of_departure: datetime.date`, `budget: int`
- Parse `VACATION_INFO_DICT` into a `VacationInfo` instance
- Extract `date_of_arrival` and `date_of_departure` to drive data gathering

### Phase 3: Gather Weather & Activity Data
- Call `call_weather_api_mocked(date, city)` for each date in the vacation range
- Call `call_activities_api_mocked(date, city)` for each date in the vacation range
- Both use `pd.date_range(start=vacation_info.date_of_arrival, end=vacation_info.date_of_departure)`
- Store results as `weather_for_dates` (list of dicts) and `activities_for_dates` (flat list of dicts)
- Display as Pandas DataFrames for manual review

### Phase 4: ItineraryAgent — Generate Initial Itinerary

**Core prompt: `ITINERARY_AGENT_SYSTEM_PROMPT`**

Components as implemented:
1. **Role**: "You are an expert travel planner for AgentsVille, specializing in creating personalized day-by-day itineraries..."
2. **Task**: 6 strict rules — weather awareness, interest matching, budget compliance, minimum 1 activity/day, no hallucinated activities, correct dates
3. **Chain-of-Thought**: "Planning Process (follow step by step)" — 6 numbered steps from interest review through JSON assembly
4. **Output format**: Two sections — `ANALYSIS` (day-by-day reasoning) then `FINAL OUTPUT` (JSON matching TravelPlan schema)
5. **Schema injection**: `TravelPlan.model_json_schema()` embedded in the prompt
6. **Context injection**:
   - `vacation_info.model_dump_json()` — full traveler details
   - `weather_for_dates` — weather forecast for all dates
   - `activities_for_dates` — all available activities

**JSON parsing**: Extracts content between `` ```json `` fences, validates with `TravelPlan.model_validate_json()`, falls back to `json_repair` on failure.

### Phase 5: Evaluate the Itinerary

7 evaluation functions (the last added in Phase 7 setup):

| Function | What it checks |
|----------|---------------|
| `eval_start_end_dates_match` | Arrival/departure dates match the plan |
| `eval_total_cost_is_accurate` | Sum of activity prices equals stated total |
| `eval_total_cost_is_within_budget` | Total cost does not exceed budget |
| `eval_itinerary_events_match_actual_events` | Every activity_id exists and details match mock data |
| `eval_itinerary_satisfies_interests` | Each traveler has at least one interest-matching activity |
| `eval_activities_and_weather_are_compatible` | No outdoor-only activities during inclement weather (LLM-powered) |
| `eval_traveler_feedback_is_incorporated` | Traveler feedback ("at least 2 activities per day") is met (LLM-powered) |

**Core prompt: `ACTIVITY_AND_WEATHER_ARE_COMPATIBLE_SYSTEM_PROMPT`**

As implemented:
1. **Role**: "You are a weather-activity compatibility evaluator"
2. **Task**: Determine if an activity should be avoided given weather conditions. Default to IS_COMPATIBLE when uncertain. Check for indoor backup options in descriptions.
3. **Output format**: `REASONING:` (step-by-step analysis) then `FINAL ANSWER:` (`IS_COMPATIBLE` or `IS_INCOMPATIBLE`)
4. **Examples**: 3 examples covering:
   - Outdoor hiking + thunderstorm → IS_INCOMPATIBLE
   - Museum art tour + rainy → IS_COMPATIBLE
   - Beach volleyball with indoor backup + rainy → IS_COMPATIBLE

Uses `CLAUDE_HAIKU` for speed (called once per activity in the itinerary).

### Phase 6: Define Tools for the Revision Agent

| Tool | Purpose | Key Parameters |
|------|---------|---------------|
| `calculator_tool` | Evaluate math expressions accurately | `input_expression: str` (e.g., `"20 + 15 + 20"`) |
| `get_activities_by_date_tool` | Retrieve activities for a specific date and city | `date: str` ("YYYY-MM-DD"), `city: str` |
| `run_evals_tool` | Run all 7 evaluation functions on an itinerary | `travel_plan: TravelPlan` (dict or Pydantic object) |
| `final_answer_tool` | Signal completion, return the final itinerary | `final_output: TravelPlan` |

The `get_activities_by_date_tool` docstring includes:
- Full description of purpose and use case
- `date` parameter: type `str`, format `"YYYY-MM-DD"`, with example
- `city` parameter: type `str`, with example
- Return type: `List[dict]` with listed fields

Tool descriptions are dynamically generated via `get_tool_descriptions_string(ALL_TOOLS)` for injection into prompts.

### Phase 7: ItineraryRevisionAgent (ReAct Loop)

**Traveler feedback**: `"I want to have at least two activities per day."` — enforced by `eval_traveler_feedback_is_incorporated`.

**Core prompt: `ITINERARY_REVISION_AGENT_SYSTEM_PROMPT`**

As implemented:
1. **Role**: "You are an itinerary revision specialist for AgentsVille"
2. **Task**: 9 explicit rules including incorporating traveler feedback, running evals first, fixing issues, using tools for lookups/calculations, and mandatory final eval before exit
3. **ReAct cycle**: Explicitly details THOUGHT → ACTION → OBSERVATION with descriptions of each step
4. **Available Tools**: Dynamically injected via `get_tool_descriptions_string(ALL_TOOLS)`
5. **ACTION format**: `{"tool_name": "[tool_name]", "arguments": {"arg1": "value1", ...}}`
6. **Mandatory rule**: "Before providing the final output, you MUST run `run_evals_tool` one final time"
7. **Exit instruction**: "When all evaluations pass, call `final_answer_tool` with the final itinerary"
8. **Context**: Vacation info, TravelPlan schema, Activity schema, weather data, available activities — all embedded

**Python ReAct loop** (`run_react_cycle` method):
- Max 15 iterations
- Each iteration: get Claude response → parse ACTION JSON → execute tool → feed OBSERVATION back
- Uses `json_repair` for robust tool-call parsing
- `final_answer_tool` triggers loop exit and TravelPlan validation
- Error messages fed back to the agent for self-correction

### Phase 8 (Bonus): Fun Narrative Summary
- Uses `narrate_my_trip()` from `project_lib.py`
- Claude generates a creative travel narrative from the final itinerary
- Optionally produces audio via OpenAI TTS (if available)

---

## Anthropic SDK vs OpenAI — Key Differences

| OpenAI Pattern | Anthropic Equivalent |
|---|---|
| `system` role in messages list | `system=` parameter on `messages.create()` |
| `response.choices[0].message.content` | `response.content[0].text` |
| `model="gpt-4.1"` | `model="claude-opus-4-6"` |
| `response_format={"type": "json_object"}` | Instruct "Return ONLY valid JSON" in prompt + `json_repair` fallback |

These differences are handled by the `do_chat_completion_anthropic` monkey-patch, which:
- Extracts system messages from the messages list into the `system=` parameter
- Ensures at least one conversation message exists
- Returns `response.content[0].text`

---

## Implementation Notes

1. **`project_lib.py` is never modified** — all customization is done via monkey-patching and the notebook
2. **JSON parsing**: Claude lacks a native JSON mode. All responses are parsed by extracting content between `` ```json `` and `` ``` `` fences, then validated with Pydantic. `json_repair` is used as a fallback for malformed JSON.
3. **Pydantic schema injection**: `TravelPlan.model_json_schema()` is injected into both the ItineraryAgent and RevisionAgent system prompts
4. **API key security**: Stored in `.env` as `ANTHROPIC_API_KEY`, loaded via `python-dotenv`
5. **Model selection**: `CLAUDE_OPUS` for complex reasoning tasks, `CLAUDE_HAIKU` for high-frequency lightweight evaluations
6. **The weather-activity compatibility eval** is the only evaluation that requires an LLM call (per activity per day)
7. **The traveler feedback eval** also uses an LLM call (`CLAUDE_OPUS`) to assess whether feedback was fully incorporated
