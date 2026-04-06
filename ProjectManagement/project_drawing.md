# Program Overview — AI-Powered Agentic Workflow for Email Router

## System Flow Diagram

```mermaid
flowchart TB
    %% ── Styling ──
    classDef inputNode fill:#1a1a2e,stroke:#e94560,color:#fff,stroke-width:2px
    classDef planNode fill:#0f3460,stroke:#53a8b6,color:#fff,stroke-width:2px
    classDef routeNode fill:#533483,stroke:#9b59b6,color:#fff,stroke-width:2px
    classDef pmNode fill:#0a6e4e,stroke:#1abc9c,color:#fff,stroke-width:2px
    classDef pgmNode fill:#b8860b,stroke:#f39c12,color:#fff,stroke-width:2px
    classDef devNode fill:#8b0000,stroke:#e74c3c,color:#fff,stroke-width:2px
    classDef evalNode fill:#4a148c,stroke:#8e44ad,color:#fff,stroke-width:2px
    classDef dualNode fill:#00695c,stroke:#26a69a,color:#fff,stroke-width:2px
    classDef outputNode fill:#1b5e20,stroke:#4caf50,color:#fff,stroke-width:2px

    %% ════════════════════════════════════════
    %% INPUTS
    %% ════════════════════════════════════════
    SPEC["Product Spec - Email Router TXT"]:::inputNode
    ENV[".env - OpenAI + Anthropic API Keys"]:::inputNode
    PROMPT["Workflow Prompt: Create complete development plan for the Email Router"]:::inputNode

    %% ════════════════════════════════════════
    %% DUAL MODEL ENGINE
    %% ════════════════════════════════════════
    subgraph DUAL["DualModelQueryEngine — Cross-Model Consensus"]
        direction LR
        D_OAI["GPT-5.4<br/>OpenAI"]:::dualNode
        D_ANT["Claude Opus 4.6<br/>Anthropic"]:::dualNode
        D_JUDGE["gpt-3.5-turbo<br/>Judge Model"]:::dualNode
        D_PICK["Best Response Selected"]:::dualNode
        D_OAI --> D_JUDGE
        D_ANT --> D_JUDGE
        D_JUDGE --> D_PICK
    end

    %% ════════════════════════════════════════
    %% PHASE 2: ORCHESTRATED WORKFLOW
    %% ════════════════════════════════════════
    subgraph WORKFLOW["PHASE 2 — Orchestrated Agentic Workflow"]
        direction TB

        AP["ActionPlanningAgent<br/>Extracts ordered workflow steps<br/>from prompt using knowledge"]:::planNode
        AP_OUT["Planned Steps:<br/>1 - Define user stories<br/>2 - Group into product features<br/>3 - Create engineering tasks"]:::planNode

        ROUTER["RoutingAgent<br/>text-embedding-3-large embeddings<br/>+ cosine similarity matching"]:::routeNode

        %% ── Product Manager Track ──
        subgraph PM_TRACK["Product Manager Track"]
            direction TB
            PM_KA["KnowledgeAugmentedPromptAgent<br/>Persona: Product Manager<br/>Knowledge: Product Spec + Story Format"]:::pmNode
            PM_EVAL["EvaluationAgent<br/>Criteria: As a user, I want X so that Y<br/>Max 3 iterations with correction loop"]:::evalNode
            PM_OUT["USER STORIES<br/>As a [type of user],<br/>I want [action or feature]<br/>so that [benefit or value]"]:::pmNode
            PM_KA -->|"worker generates response"| PM_EVAL
            PM_EVAL -->|"FAIL: generate correction<br/>instructions and retry"| PM_KA
            PM_EVAL -->|"PASS"| PM_OUT
        end

        %% ── Program Manager Track ──
        subgraph PGM_TRACK["Program Manager Track"]
            direction TB
            PGM_KA["KnowledgeAugmentedPromptAgent<br/>Persona: Program Manager<br/>Knowledge: Feature Grouping Rules"]:::pgmNode
            PGM_EVAL["EvaluationAgent<br/>Criteria: Feature Name, Description,<br/>Key Functionality, User Benefit<br/>Max 3 iterations with correction loop"]:::evalNode
            PGM_OUT["PRODUCT FEATURES<br/>Feature Name - Description<br/>Key Functionality<br/>User Benefit"]:::pgmNode
            PGM_KA -->|"worker generates response"| PGM_EVAL
            PGM_EVAL -->|"FAIL: generate correction<br/>instructions and retry"| PGM_KA
            PGM_EVAL -->|"PASS"| PGM_OUT
        end

        %% ── Development Engineer Track ──
        subgraph DEV_TRACK["Development Engineer Track"]
            direction TB
            DEV_KA["KnowledgeAugmentedPromptAgent<br/>Persona: Development Engineer<br/>Knowledge: Task Structuring Rules"]:::devNode
            DEV_EVAL["EvaluationAgent<br/>Criteria: Task ID, Title, Related Story,<br/>Description, Acceptance Criteria,<br/>Estimated Effort, Dependencies<br/>Max 3 iterations with correction loop"]:::evalNode
            DEV_OUT["ENGINEERING TASKS<br/>Task ID - Task Title<br/>Related User Story - Description<br/>Acceptance Criteria<br/>Estimated Effort - Dependencies"]:::devNode
            DEV_KA -->|"worker generates response"| DEV_EVAL
            DEV_EVAL -->|"FAIL: generate correction<br/>instructions and retry"| DEV_KA
            DEV_EVAL -->|"PASS"| DEV_OUT
        end

        %% ── Internal Flow ──
        AP --> AP_OUT
        AP_OUT --> ROUTER
        ROUTER -->|"Step 1: highest cosine<br/>similarity to PM description"| PM_TRACK
        ROUTER -->|"Step 2: highest cosine<br/>similarity to PgM description"| PGM_TRACK
        ROUTER -->|"Step 3: highest cosine<br/>similarity to Dev description"| DEV_TRACK
        PM_OUT -->|"user stories context<br/>feeds into features"| PGM_KA
        PM_OUT -->|"user stories context<br/>feeds into tasks"| DEV_KA
        PGM_OUT -->|"features context<br/>feeds into tasks"| DEV_KA
    end

    %% ════════════════════════════════════════
    %% FINAL OUTPUT
    %% ════════════════════════════════════════
    FINAL["FINAL DEVELOPMENT PLAN<br/>User Stories + Product Features + Engineering Tasks"]:::outputNode

    %% ════════════════════════════════════════
    %% TOP-LEVEL CONNECTIONS
    %% ════════════════════════════════════════
    SPEC --> AP
    SPEC --> PM_KA
    ENV --> DUAL
    ENV --> WORKFLOW
    PROMPT --> AP
    DUAL -.->|"dual-model evaluation<br/>and planning calls"| PM_EVAL
    DUAL -.->|"dual-model<br/>evaluation"| PGM_EVAL
    DUAL -.->|"dual-model<br/>evaluation"| DEV_EVAL
    DUAL -.->|"dual-model<br/>step extraction"| AP
    PM_OUT --> FINAL
    PGM_OUT --> FINAL
    DEV_OUT --> FINAL
```

---

## Phase 1 — Agent Library Architecture

```mermaid
flowchart LR
    classDef simple fill:#2c3e50,stroke:#3498db,color:#fff,stroke-width:2px
    classDef medium fill:#1a3a4a,stroke:#1abc9c,color:#fff,stroke-width:2px
    classDef complex fill:#3c1361,stroke:#9b59b6,color:#fff,stroke-width:2px
    classDef infra fill:#00695c,stroke:#26a69a,color:#fff,stroke-width:2px

    subgraph LIB["base_agents.py — 7 Agent Classes + DualModelQueryEngine"]
        direction TB

        subgraph SIMPLE["Simple Agents — Single LLM Call"]
            direction LR
            A1["DirectPromptAgent<br/>No system prompt<br/>Pure LLM response"]:::simple
            A2["AugmentedPromptAgent<br/>System prompt sets persona<br/>LLM + personality"]:::simple
            A3["KnowledgeAugmentedPromptAgent<br/>Persona + injected knowledge<br/>Constrained to provided facts"]:::simple
        end

        subgraph RETRIEVAL["Retrieval Agent — Embedding + Chunking"]
            A4["RAGKnowledgePromptAgent<br/>Chunks text, embeds with<br/>text-embedding-3-large,<br/>cosine similarity retrieval,<br/>answers from best chunk"]:::medium
        end

        subgraph COMPOSITE["Composite Agents — Multi-Step Logic"]
            direction LR
            A5["EvaluationAgent<br/>Iterative evaluate-correct loop<br/>Worker response, judge pass/fail,<br/>generate corrections, retry<br/>up to max_interactions"]:::complex
            A6["RoutingAgent<br/>Embed prompt + agent descriptions,<br/>cosine similarity ranking,<br/>dispatch to best-matched agent"]:::complex
            A7["ActionPlanningAgent<br/>Extract ordered steps<br/>from natural language prompt<br/>using domain knowledge"]:::complex
        end

        subgraph ENGINE["Dual-Model Engine"]
            DM["DualModelQueryEngine<br/>GPT-5.4 + Opus 4.6<br/>gpt-3.5-turbo judge<br/>selects best response"]:::infra
        end
    end

    A1 -.-> A2
    A2 -.-> A3
    A3 -.-> A4
    A3 -->|"used as worker by"| A5
    A3 -->|"routed by"| A6
    A7 -->|"plans steps for"| A6
    DM -->|"enhances complex calls in"| A5
    DM -->|"enhances planning in"| A7
```

---

## Evaluation Agent — Internal Correction Loop Detail

```mermaid
flowchart TD
    classDef worker fill:#0a6e4e,stroke:#1abc9c,color:#fff,stroke-width:2px
    classDef eval fill:#4a148c,stroke:#8e44ad,color:#fff,stroke-width:2px
    classDef decision fill:#b8860b,stroke:#f39c12,color:#fff,stroke-width:2px
    classDef output fill:#1b5e20,stroke:#4caf50,color:#fff,stroke-width:2px

    START["Initial Prompt"]
    WORKER["agent_to_evaluate.respond<br/>Worker agent generates response"]:::worker
    JUDGE["EvaluationAgent._call_llm<br/>Judge: Does response meet criteria?<br/>temperature=0"]:::eval
    CHECK{"Starts with YES<br/>or PASS or<br/>meets the criteria?"}:::decision
    CORRECT["EvaluationAgent._call_llm<br/>Generate correction instructions<br/>temperature=0"]:::eval
    FEEDBACK["Build corrected prompt:<br/>original + response +<br/>correction instructions"]
    MAXCHECK{"Reached<br/>max_interactions?"}:::decision
    DONE["Return dict:<br/>final_response<br/>evaluation<br/>iterations<br/>success"]:::output

    START --> WORKER
    WORKER --> JUDGE
    JUDGE --> CHECK
    CHECK -->|"YES"| DONE
    CHECK -->|"NO"| CORRECT
    CORRECT --> FEEDBACK
    FEEDBACK --> MAXCHECK
    MAXCHECK -->|"NO: iterate"| WORKER
    MAXCHECK -->|"YES: stop"| DONE
```

---

## Routing Agent — Semantic Dispatch Detail

```mermaid
flowchart TD
    classDef embed fill:#533483,stroke:#9b59b6,color:#fff,stroke-width:2px
    classDef compare fill:#0f3460,stroke:#53a8b6,color:#fff,stroke-width:2px
    classDef agent fill:#0a6e4e,stroke:#1abc9c,color:#fff,stroke-width:2px

    INPUT["Incoming Step Text"]
    EMB_INPUT["get_embedding - input<br/>text-embedding-3-large"]:::embed

    subgraph AGENTS["Registered Agent Descriptions"]
        direction LR
        EMB_PM["PM embedding:<br/>define personas and<br/>user stories only"]:::embed
        EMB_PGM["PgM embedding:<br/>group user stories<br/>into product features"]:::embed
        EMB_DEV["Dev embedding:<br/>create detailed<br/>engineering tasks"]:::embed
    end

    COS1["cosine_similarity<br/>input vs PM"]:::compare
    COS2["cosine_similarity<br/>input vs PgM"]:::compare
    COS3["cosine_similarity<br/>input vs Dev"]:::compare

    BEST{"Select agent with<br/>highest similarity score"}:::compare

    FUNC_PM["PM support function"]:::agent
    FUNC_PGM["PgM support function"]:::agent
    FUNC_DEV["Dev support function"]:::agent

    INPUT --> EMB_INPUT
    EMB_INPUT --> COS1
    EMB_INPUT --> COS2
    EMB_INPUT --> COS3
    EMB_PM --> COS1
    EMB_PGM --> COS2
    EMB_DEV --> COS3
    COS1 --> BEST
    COS2 --> BEST
    COS3 --> BEST
    BEST -->|"PM wins"| FUNC_PM
    BEST -->|"PgM wins"| FUNC_PGM
    BEST -->|"Dev wins"| FUNC_DEV
```

---

## Data Flow Summary

```mermaid
flowchart LR
    classDef data fill:#263238,stroke:#607d8b,color:#eceff1,stroke-width:2px
    classDef process fill:#1a237e,stroke:#5c6bc0,color:#fff,stroke-width:2px

    SPEC["Product Spec TXT"]:::data
    STORIES["User Stories"]:::data
    FEATURES["Product Features"]:::data
    TASKS["Engineering Tasks"]:::data
    PLAN["Final Dev Plan"]:::data

    PM["PM Agent + Eval Loop"]:::process
    PGM["PgM Agent + Eval Loop"]:::process
    DEV["Dev Agent + Eval Loop"]:::process

    SPEC --> PM
    PM --> STORIES
    SPEC --> PGM
    STORIES --> PGM
    PGM --> FEATURES
    SPEC --> DEV
    STORIES --> DEV
    FEATURES --> DEV
    DEV --> TASKS
    STORIES --> PLAN
    FEATURES --> PLAN
    TASKS --> PLAN
```
