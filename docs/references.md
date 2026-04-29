# References

## Key Concepts

- **Transformers in Control**: Using the Transformer architecture (Attention mechanisms) to process sequential sensor data and predict actions, replacing traditional RNNs/LSTMs.
- **RoPE (Rotary Position Embeddings)**: Position encoding method used in Axis V2 for better sequence modeling. [Paper](https://arxiv.org/abs/2104.09864)
- **DINOv2**: Self-supervised vision transformer used as the image backbone for geometrically consistent features. [Paper](https://arxiv.org/abs/2304.07193)
- **Action Chunking**: Predicting multiple future actions in a single forward pass for smoother control. [Inspiration: ACT](https://arxiv.org/abs/2304.13705)

## V2 Representations

- **SE(3) Lie Group**: Axis V2 uses SE(3) for rigid body motion representation using **PyPose**.
  - Poses: 13D full matrix (flattened rotation matrix + translation + gripper)
  - Actions: 7D twist (angular velocity + linear velocity + gripper delta)
  - [PyPose: A Library for Robot Learning with Lie Theory](https://pypose.org/)
  - [Murray et al., "A Mathematical Introduction to Robotic Manipulation"](https://www.cse.lehigh.edu/~trink/Courses/RoboticsII/reading/murray-li-sastry-94-complete.pdf)

- **6D Rotation Representation**: Zhou et al.'s continuous rotation representation.
  - [Paper: On the Continuity of Rotation Representations in Neural Networks](https://arxiv.org/abs/1812.07035)

## Datasets

- **Open X-Embodiment (RTX)**: Large-scale robotic dataset from many embodiments. [Website](https://robotics-transformer-x.github.io/)
- **Fractal**: Manipulation tasks dataset within the RTX collection.
- **DROID**: A large-scale, in-the-wild robot manipulation dataset for generalist agents. [Website](https://droid-dataset.github.io/)

## Related Architectures

- **RT-1**: Transformer-based robot control with tokenized images. Axis shares the Transformer philosophy but uses SE(3) representations.
- **RT-2**: Vision-Language-Action (VLA) model.
- **ACT (Action Chunking with Transformers)**: Inspiration for action chunking approach. [Paper](https://arxiv.org/abs/2304.13705)

