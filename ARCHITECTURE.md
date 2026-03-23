# ACARA Project Architecture & Flow

This document details the **Adaptive Context-Aware Retrieval Architecture (ACARA)** flow.

## 🛠 System Flow Diagram

```mermaid
graph TD
    subgraph User Interaction
        User[User Query] --> QE[Query Encoder]
    end

    subgraph Memory & Retrieval
        QE --> VM[(Vector Memory)]
        VM --> CAG{Context Awareness Gate}
    end

    subgraph Adaptive Retrieval Controller
        ARC[Adaptive Retrieval Controller - ARC]
        ARC -.->|similarity_threshold| CAG
        ARC -.->|top_k| VM
        ARC -.->|chunk_size / overlap| DCM
    end

    subgraph Gating Logic
        CAG -->|Similarity Pass| Cov{Coverage Check}
        Cov -->|LLM Verified| Fresh{Freshness Check}
    end

    subgraph Dynamic Routing
        Fresh -->|STALENESS / WEAK| EKS[External Knowledge Source - Tavily]
        CAG -->|LOW SIMILARITY| EKS
        Cov -->|LOW COVERAGE| EKS
        
        Fresh -->|STRONG| CB[Context Builder]
    end

    subgraph Knowledge Processing
        EKS --> DCM[Dynamic Chunking Module]
        DCM --> Cred{Credibility Scoring}
        Cred -->|Validated| MU[Memory Update - Vector Store]
        Cred --> CB
    end

    subgraph LLM Synthesis
        CB --> GEN[Generator Model - GPT-4o-mini]
        GEN --> VAL{Critic / Validator}
        VAL -->|Hallucination Warn| FO[Final Output]
        VAL -->|Clean| FO
    end

    %% Feedback loop
    CAG -.->|WEAK Signal| ARC
    CAG -.->|STRONG Signal| ARC
```

## 🧩 Core Components

### 1. ARC (Adaptive Retrieval Controller)
The "brain" of the architecture. It dynamically adjusts retrieval parameters (threshold, top_k, chunking) based on pipeline feedback signals.

### 2. Context Awareness Gate
A multi-dimensional filter that evaluates retrieved chunks for:
- **Similarity**: Vector distance against dynamic ARC threshold.
- **Coverage**: Semantic relevance determined by structured LLM grading.
- **Freshness**: Date-based filtering to avoid stale information.

### 3. Dynamic Chunking Module
Adjusts chunk size and overlap based on query complexity, ensuring optimal retrieval for both short and long queries.

### 4. Credibility Scoring
Validates external knowledge before it is used for generation or stored in long-term memory.

### 5. Critic / Validator
A final verification stage that flags hallucinations or unsupported claims before the user sees the output.
