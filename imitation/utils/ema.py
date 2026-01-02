import torch
import copy

class EMA:
    """
    Exponential Moving Average for model parameters.
    Maintains a shadow copy of the model and updates it using:
    shadow_variable -= (1 - decay) * (shadow_variable - variable)
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.requires_grad_(False)
        self.shadow.eval()

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            ssd = self.shadow.state_dict()
            for k in msd:
                if msd[k].dtype.is_floating_point:
                    # Skip if shape mismatch (e.g. dynamic buffers like latent_queue)
                    if msd[k].shape != ssd[k].shape:
                        continue
                    ssd[k].copy_(self.decay * ssd[k] + (1. - self.decay) * msd[k])

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)
