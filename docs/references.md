# References

## Key Concepts

- **Transformers in Control**: Using the Transformer architecture (Attention mechanisms) to process sequential sensor data and predict actions, replacing traditional RNNs/LSTMs.
- **Latent Dynamics**: Learning a compact state representation that evolves over time, allowing the model to reason about the "state of the world" beyond just the immediate observation.
- **Token Learner**: A method to adaptively select a small number of tokens from an image to represent the visual content, reducing computational complexity for Transformers. [Paper: TokenLearner: What Can 8 Learned Tokens Do for Images and Video?](https://arxiv.org/abs/2106.11297)

## Datasets

- **Open X-Embodiment (RTX)**: A large-scale robotic dataset containing data from many different robot embodiments. Axis is designed to leverage this diversity via its Robot Adapter architecture. [Website](https://robotics-transformer-x.github.io/)
- **Fractal**: A specific dataset often used within the RTX collection, focusing on manipulation tasks.

## Related Architectures

- **RT-1 (Robotics Transformer 1)**: A Transformer-based model for robot control that tokenizes images and instructions. Axis shares the philosophy of using Transformers but introduces Latent Memory and specialized Adapters.
- **RT-2**: A Vision-Language-Action (VLA) model.
