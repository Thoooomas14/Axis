# Axis Model Design

The Axis model is designed to be a general-purpose, multi-robot control policy that learns from diverse datasets. It treats robot control as a sequence modeling problem, predicting the next action based on a history of observations and a semantic goal.

## Architecture Overview

The model follows a modular architecture composed of:
1.  **Encoders**: Process raw inputs (Vision, Goal, Proprioception).
2.  **Bottleneck**: Compresses high-dimensional visual data into a compact set of tokens.
3.  **Latent Memory**: Maintains a running context of the episode (optional/disabled by default).
4.  **Transformer Backbone**: Fuses all modalities and attends to relevant features.
5.  **Decoders**: Translates transformer outputs into robot actions.

**Window-Based Processing**:
Axis processes a sliding window of observations (size $W=8$) at each step. This allows the model to attend to immediate temporal gradients and short-term history directly in the transformer.

```mermaid
graph TD
    subgraph Inputs
        Img[Image Window (B, W, C, H, W)]
        Prop[Proprio Window (B, W, 8)]
        Goal[Goal Embedding (B, 64)]
    end

    subgraph Encoders
        VE[Vision Encoder]
        GE[Goal Encoder]
        PE[Proprio Encoder]
    end

    subgraph Bottleneck
        TL[Token Learner]
    end

    subgraph Memory
        LQ[Latent Queue]
        LP[Latent Projection]
    end

    subgraph Backbone
        T[Axis Transformer]
    end

    subgraph Outputs
        AD[Action Decoder]
        RD[Requery Decoder]
        L_Out[Latent Update]
    end

    Goal --> GE
    GE --> VE
    Img --> VE --> TL
    Prop --> PE
    
    LQ --> LP
    
    TL --> T
    PE --> T
    LP --> T
    
    T --> AD
    T --> RD
    T --> L_Out
    L_Out -.-> LQ
```

## Component Details

### 1. Goal Encoder
-   **Input**: A **64-dimensional** semantic vector.
-   **Composition**: The goal vector is a random projection of:
    -   **Task Type** (One-hot: Pick, Place, Move)
    -   **Start Pose** (8D: 3 Pos + 4 Quat + 1 Gripper)
    -   **End Pose / POI** (8D: 3 Pos + 4 Quat + 1 Gripper)
-   **Function**: Projects the goal embedding into a conditioning vector used by the Vision Encoder.

### 2. Vision Encoder & Token Learner
-   **Vision Encoder**: A CNN-based backbone that extracts spatial features from the input image. It is conditioned on the goal embedding via FiLM to focus on relevant objects.
-   **Token Learner**: Reduces the spatial feature map (H x W x C) into a small, fixed number of tokens (8 per image). This significantly reduces the computational cost for the transformer.

### 3. Latent Memory (Queue)
-   **Concept**: Axis maintains a "Latent Queue" of the past $N$ states to handle long-term history.
-   **Status**: Currently disabled by default for initial training phases to focus on immediate window-based control.

### 4. Axis Transformer
-   **Type**: Transformer Encoder (standard attention mechanism).
-   **Input**: A concatenated sequence of:
    -   **Latent Tokens**: Projected history from the queue (if enabled).
    -   **Context Tokens**: Interleaved sequence of `[Proprio_t, Vision_t]` for each step $t$ in the window $W$.
-   **Role**: Performs sensor fusion and reasoning. It allows the proprioceptive state to attend to visual features and historical context to determine the best action.

### 5. Decoders
-   **Action Decoder**: Projects the transformer's output token (corresponding to the last step) into the **8D Action Space** (Position + Rotation(Quat) + Gripper).
-   **Requery Decoder**: Predicts a binary logit indicating if the agent needs a new high-level instruction (e.g., subtask complete).

## Data Flow

1.  **Observation**: The data loader streams a window of $W=8$ images and proprioceptive states.
2.  **Encoding**:
    -   Goal (64D) is encoded.
    -   Images are processed and tokenized (conditioned on goal).
    -   Proprioception (8D) is projected to embedding space.
3.  **Fusion**: All tokens are fed into the Transformer.
4.  **Prediction**:
    -   The output corresponding to the **last proprioception token** in the window is decoded into the next **Action** (8D).
    -   A **Requery** signal is predicted.
