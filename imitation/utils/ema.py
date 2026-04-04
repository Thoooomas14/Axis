import torch

class EMA:
    """
    Exponential Moving Average for model parameters.
    Maintains a shadow copy of the model parameters and updates it using:
    shadow_variable = decay * shadow_variable + (1 - decay) * variable
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}

        # Initialize the shadow with a detached clone of the model's state_dict.
        # This completely avoids the PickleError caused by TorchScript/JIT deepcopying.
        for k, v in model.state_dict().items():
            self.shadow[k] = v.clone().detach()

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k in msd:
                if msd[k].dtype.is_floating_point:
                    # Skip if shape mismatch (e.g. dynamic buffers like latent_queue)
                    if k in self.shadow and msd[k].shape != self.shadow[k].shape:
                        continue
                    self.shadow[k].copy_(self.decay * self.shadow[k] + (1.0 - self.decay) * msd[k])
                else:
                    # Directly copy non-floating point tensors (like integer step counters)
                    if k in self.shadow:
                        self.shadow[k].copy_(msd[k])

    def state_dict(self):
        # Return the dictionary directly, as it acts as the state_dict
        return self.shadow

    def load_state_dict(self, state_dict):
        # Safely copy values into the existing shadow tensors to maintain device placement
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)
            else:
                self.shadow[k] = v.clone().detach()

    def apply_to(self, model):
        """
        Utility to copy the EMA weights back into a model for evaluation.
        """
        model.load_state_dict(self.shadow)