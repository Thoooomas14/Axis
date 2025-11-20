import torch
import torch.nn as nn
from .encoders import ProprioEncoder
from .decoders import ActionDecoder

class RobotAdapter(nn.Module):
    """
    Adapter for a specific robot configuration.
    Contains the ProprioEncoder and ActionDecoder for that robot.
    """
    def __init__(self, proprio_dim, action_dim, embed_dim=256):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        self.proprio_encoder = ProprioEncoder(
            input_dim=proprio_dim,
            output_dim=embed_dim
        )

        self.action_decoder = ActionDecoder(
            embed_dim=embed_dim,
            action_dim=action_dim
        )

    def encode_proprio(self, proprio):
        """
        Args:
            proprio: (B, proprio_dim)
        Returns:
            (B, embed_dim)
        """
        return self.proprio_encoder(proprio)

    def decode_action(self, transformer_output):
        """
        Args:
            transformer_output: (B, embed_dim)
        Returns:
            (B, action_dim)
        """
        return self.action_decoder(transformer_output)
