# Axis Model Design

The Axis model is designed to be a general-purpose, multi-robot control policy that learns from diverse datasets. It treats robot control as a sequence modeling problem, predicting the next action based on a history of observations and a semantic goal.

## Architecture Overview

The model follows a modular architecture composed of:
1.  **Encoders**: Process raw inputs (Vision, Goal, Proprioception).
2.  **Bottleneck**: Compresses high-dimensional visual data into a compact set of tokens.
3.  **Latent Memory**: Maintains a running context of the episode.
4.  **Transformer Backbone**: Fuses all modalities and attends to relevant features.
5.  **Decoders/Adapters**: Translates transformer outputs into robot-specific actions.

**Window-Based Processing**:
Unlike typical frame-by-frame policies, Axis processes a sliding window of observations (size $W$) at each step. This allows the model to attend to immediate temporal gradients and short-term history directly in the transformer, while long-term history is handled by the Latent Memory.

```mermaid
graph TD
    subgraph Inputs
        Img[Image Window (B, W, C, H, W)]
        Prop[Proprio Window (B, W, D)]
        Goal[Goal Embedding]
    end

    subgraph Encoders
        VE[Vision Encoder]
        GE[Goal Encoder]
        RA_Enc[Robot Adapter Encoder]
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
        RA_Dec[Robot Adapter Decoder]
        RD[Requery Decoder]
        L_Out[Latent Update]
    end

    Goal --> GE
    GE --> VE
    Img --> VE --> TL
    Prop --> RA_Enc
    
    LQ --> LP
    
    TL --> T
    RA_Enc --> T
    LP --> T
    
    T --> RA_Dec
    T --> RD
    T --> L_Out
    L_Out -.-> LQ
```

## Component Details

### 1. Goal Encoder
- **Input**: Pre-computed semantic embeddings (e.g., from a language model like CLIP or T5).
- **Function**: Projects the goal embedding into a conditioning vector used by the Vision Encoder.

### 2. Vision Encoder & Token Learner
- **Vision Encoder**: A CNN-based backbone (e.g., ResNet-like) that extracts spatial features from the input image. It is conditioned on the goal embedding via FiLM (Feature-wise Linear Modulation) or simple concatenation to focus on relevant objects.
- **Token Learner**: Reduces the spatial feature map (H x W x C) into a small, fixed number of tokens (K x C). This significantly reduces the computational cost for the transformer and forces the model to learn a compact representation of the visual scene.

### 3. Latent Memory (Queue)
- **Concept**: Instead of feeding the entire history of images to the transformer (which is expensive), Axis maintains a "Latent Queue" of the past $N$ states.
- **Mechanism**: At each step, the model outputs a `next_latent` vector. This vector is pushed into the queue, and the oldest vector is popped.
- **Benefit**: Allows the model to reason about temporal dynamics and short-term history without processing full image sequences every step.

### 4. Axis Transformer
- **Type**: Transformer Encoder (standard attention mechanism).
- **Input**: A concatenated sequence of:
    - **Latent Tokens**: Projected history from the queue.
    - **Context Tokens**: Interleaved sequence of `[Proprio_t, Vision_t]` for each step $t$ in the window $W$.
- **Role**: Performs sensor fusion and reasoning. It allows the proprioceptive state to attend to visual features and historical context to determine the best action.

### 5. Robot Adapters
- **Problem**: Different robots have different physical structures (proprioception dimensions) and control interfaces (action dimensions).
- **Solution**: Robot Adapters act as translators.
    - **Encoder**: Projects robot-specific proprioception (e.g., 6-DOF pose + gripper) into the shared embedding space.
    - **Decoder**: Projects the transformer's output token back into the robot-specific action space.
- **Usage**: Allows a single core Axis model to be trained on mixed datasets from different robots (e.g., UR3e, WidowX, Franka).

### 6. Requery Mechanism
- **Purpose**: To signal when a sub-task is complete or if the agent is "confused" and needs a new high-level instruction.
- **Output**: A binary logit (probability) indicating the need to re-query the high-level planner.

## Data Flow

1.  **Observation**: The robot observes a window of $W$ images and proprioceptive states. A high-level goal is provided.
2.  **Encoding**:
    - Goal is encoded.
    - Images are processed and tokenized (conditioned on goal).
    - Proprioception is projected to embedding space.
3.  **Context Retrieval**: The current state of the Latent Queue is retrieved and projected.
4.  **Fusion**: All tokens (Latent + Windowed Context) are fed into the Transformer.
5.  **Prediction**:
    - The output corresponding to the **last proprioception token** in the window is decoded into the next **Action**.
    - A **Next Latent** state is generated to update the memory.
    - A **Requery** signal is predicted.
6.  **Update**: The Latent Queue is updated with the new latent state for the next timestep.
