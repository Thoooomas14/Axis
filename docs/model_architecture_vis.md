# Axis V2 Model Architecture

This document visualizes the data flow and structure of the Axis V2 model.

## High-Level Data Flow

```mermaid
graph TD
    subgraph Inputs
        IMG[("Images<br/>(B, W, 3, 224, 224)")]
        PROP[("Proprioception<br/>(B, W, 13)<br/>[R_flat(9), Pos, Gripper]")]
        GOAL[("Goal Embed<br/>(B, 64)")]
    end

    subgraph Encoders
        VE[("VisionEncoder<br/>ResNet-18")]
        PE[("ProprioEncoder<br/>13D → 128D")]
        GE[("GoalEncoder<br/>64D → 128D")]
        TL[("TokenLearner<br/>1 token/frame")]
    end

    subgraph Token_Construction
        CAT[("Per-Frame Concatenation<br/>[Proprio | Vision | Goal]<br/>128 + 768 + 128 = 1024D")]
    end

    subgraph Backbone
        TR[("AxisTransformer<br/>512D, RoPE, 4 layers")]
    end

    subgraph Decoders
        LT[("Last Token Bottleneck<br/>Extract output_tokens[:, -1, :]")]
        AD[("ActionDecoder<br/>MLP: 512D → ChunkSize*7D")]
        RD[("RequeryDecoder<br/>MLP: 512D → 1D")]
    end

    subgraph Outputs
        ACT[("Future Trajectory<br/>(B, ChunkSize, 7)")]
        REQ[("Confidence<br/>(B, 1)")]
    end

    %% Connections
    IMG --> VE --> TL
    TL -->|"(B, W, 256)"| VTOK[("Vision Tokens")]
    
    PROP --> PE -->|"(B, W, 128)"| PTOK[("Proprio Tokens")]
    
    GOAL --> GE -->|"(B, 128)"| GTOK[("Goal Token")]
    GTOK -->|"Repeat W times"| GEXP[("(B, W, 128)")]
    
    PTOK --> CAT
    VTOK --> CAT
    GEXP --> CAT
    CAT -->|"(B, W, 512)"| TR
    
    TR -->|"(B, W, 512)"| OUT[("Output Tokens")]
    
    OUT --> LT
    LT --> AD
    LT --> RD
    AD --> ACT
    RD --> REQ

    classDef tensor fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef component fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,rx:10,ry:10;
    classDef logic fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;

    class IMG,PROP,GOAL,VTOK,PTOK,GTOK,GEXP,OUT,ACT,REQ tensor;
    class VE,PE,GE,TL,TR,AD,RD component;
    class CAT logic;
```

## Key V2 Architecture Changes

| Component | V1 | V2 (Pure Chunking) |
|-----------|----|----|
| Proprioception | 7D (rotvec + pos + grip) | **13D** (R_flat + pos + grip) |
| Action Output | 10D delta pose | **7D twist** (ω, v, grip) |
| Token Structure | Interleaved P, V tokens | **Concatenated** per frame |
| Output Shape | (B, W, 7) sequential | **(B, ChunkSize, 7)** from last token |
| Requery | Binary logit | **Confidence** [0, 1] |

## Token Concatenation

Each timestep produces a single 512D token:

```
┌──────────────────────────────────────────────┐
│  Proprio (128D)  │  Vision (768D)  │  Goal (128D)  │
└──────────────────────────────────────────────┘
                    512D total
```

The transformer sees a sequence of W=8 such tokens.

## Vision Pipeline

```mermaid
graph LR
    I[("Images (B, W, 3, 224, 224)")] -->|Flatten B*W| VE[("ResNet-18")]
    VE -->|"(B*W, 256, H', W')"| FM[("Feature Map")]
    FM --> TL[("Token Learner")]
    TL -->|"(B*W, 1, 256)"| T[("Vision Tokens")]
    T -->|Reshape| OUT[("(B, W, 256)")]
```

## Action Chunking

V2 predicts a **full window of actions** in one forward pass:

```python
pred_action, confidence = model(images, proprio, goal)
# pred_action: (B, ChunkSize, 7) - Chunk of future twist actions
# confidence: (B, 1) - model confidence for this trajectory

# At inference, use temporal smoothing
ensembler = ActionChunkEnsembler(chunk_size=ChunkSize)
action = ensembler.update(pred_action)  # Smoothed action
```

## Decoder Details

### Action Decoder

Predicts the entire trajectory from the **last transformer token**:
```python
# Based on output_tokens[:, -1, :] (512D)
trajectory = MLP(last_token)  # 512D → 10 * 7D
pred_action = trajectory.reshape(B, 10, 7)
```

Output is clamped by `SafeActionDecoder`:
- Rotation: ±0.2 rad/step
- Translation: ±0.05 m/step

### Requery Decoder (Confidence)

Simplified to a standard MLP acting on the last token:
```python
# Based on output_tokens[:, -1, :] (512D)
confidence = sigmoid(MLP(last_token))  # 512D → 1D
```

During training, target = `exp(-action_loss / temperature)`.

## Dimension Summary

| Tensor | Shape | Description |
|--------|-------|-------------|
| Input Images | `(B, W, 3, 224, 224)` | RGB images |
| Input Proprio | `(B, W, 13)` | 13D SE(3) poses |
| Input Goal | `(B, 64)` | Semantic goal |
| Vision Tokens | `(B, W, 256)` | 1 token per frame |
| Proprio Tokens | `(B, W, 128)` | Encoded proprio |
| Goal Tokens | `(B, W, 128)` | Repeated goal |
| Transformer Input | `(B, W, 512)` | Concatenated tokens |
| Action Output | `(B, ChunkSize, 7)` | 7D twists |
| Confidence Output | `(B, 1)` | Single scalar |