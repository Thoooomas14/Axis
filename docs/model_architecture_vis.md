# Axis V1 Model Architecture

This document visualizes the data flow and structure of the compiled AxisModel based on `src/models/axis_v1.py` and `imitation/train.py`.

## High-Level Data Flow

``` mermaid
graph TD
    subgraph Inputs
        IMG[("Images<br/>(B, Window, 3, 128, 128)")]
        PROP[("Proprioception<br/>(B, Window, 8)<br/>[Pos, Quat, Gripper]")]
        GOAL[("Goal Embed<br/>(B, 64)")]
    end

    subgraph Encoders
        GE[("GoalEncoder<br/>(MLP)")]
        VE[("VisionEncoder<br/>(ResNet/CNN)")]
        PE[("ProprioEncoder<br/>(MLP)")]
        TL[("TokenLearner<br/>(Spatial Attention)")]
    end

    subgraph Context_Construction
        CAT[("Concatenate & Interleave<br/>[P_t, V_t^1...V_t^K]")]
        LQ[("Latent Queue<br/>(Memory)")]
        LP[("LatentProjection")]
    end

    subgraph Backbone
        TR[("AxisTransformer<br/>(Self-Attention layers)")]
    end

    subgraph Decoders
        READ[("Readout Selection<br/>Last Proprio Token")]
        AD[("ActionDecoder<br/>(MLP)")]
        RD[("RequeryDecoder<br/>(MLP)")]
        LPROJ[("Latent Projection Out")]
    end

    subgraph Outputs
        ACT[("Diff_Action<br/>(B, 8)")]
        REQ[("Requery Logit<br/>(B, 1)")]
        NL[("Next Latent<br/>(B, 256)")]
    end

    %% Connections
    GOAL --> GE -->|"(B, 256)"| COND[("Conditioning")]
    
    IMG --> VE
    COND -.-> VE
    VE -->|"(B*W, 256, H, W)"| TL
    TL -->|"(B, W, 8, 256)"| VTOK[("Vision Tokens")]

    PROP --> PE -->|"(B, W, 1, 256)"| PTOK[("Proprio Tokens")]

    PTOK --> CAT
    VTOK --> CAT
    CAT -->|"(B, W*(1+8), 256)"| CTX[("Context Seq")]

    LQ --> LP -->|"(B, T_mem, 256)"| MEM[("Memory Tokens")]
    
    CTX --> MERGE[("Concat Input")]
    MEM --> MERGE
    MERGE --> TR
    TR -->|"(B, SeqLen, 256)"| OUT[("Output Tokens")]

    OUT --> READ -->|"(B, 256)"| LAST[("Last P Token")]

    LAST --> AD --> ACT
    LAST --> RD --> REQ
    LAST --> LPROJ --> NL
    NL -.->|Update| LQ

    classDef tensor fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef component fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,rx:10,ry:10;
    classDef logic fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;

    class IMG,PROP,GOAL,COND,VTOK,PTOK,CTX,MEM,OUT,LAST,ACT,REQ,NL tensor;
    class GE,VE,PE,TL,TR,AD,RD,LPROJ,LQ,LP component;
    class CAT,READ,MERGE logic;
```

## Detailed Block Diagrams

### 1. Vision Pipeline

`Images -> VisionEncoder -> TokenLearner -> Tokens`

``` mermaid
graph LR
    I[("Images (B, W, 3, 128, 128)")] -->|Flatten B*W| VE[("Vision Encoder")]
    C[("Goal Cond (B, 256)")] -.->|Tile & Concat| VE
    VE -->|"(B*W, 256, H', W')"| FM[("Feature Map")]
    FM --> TL[("Token Learner")]
    TL -->|"(B*W, K=8, 256)"| T[("Vision Tokens")]
```

### 2. Sequence Structure (The "Axis" of Time)

The Transformer sees a sequence constructed by interleaving Proprioception and Vision tokens for each timestep in the window.

If Window Size `W=2` and Vision Tokens `K=3`: Sequence: `[P_0, V_0^1, V_0^2, V_0^3, P_1, V_1^1, V_1^2, V_1^3]`

-   **P_t**: Proprioception Embedding at time t
-   **V_t\^k**: k-th Vision Token at time t

### 3. Readout Logic

We specifically extract the token corresponding to the **current timestep's proprioception**. Since the model predicts the *next* action based on the history up to *now*, we use the embedding of the most recent proprioception state (at index `(W-1) * stride`) effectively as the "query" for the transformer's attention mechanism to generate the next action.

``` python
# From src/models/axis_v1.py
stride = 1 + K
readout_idx = (W - 1) * stride
last_token = output_tokens[:, readout_idx, :]
```