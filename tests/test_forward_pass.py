import torch
from src.models.axis import AxisModel

def main():
    config = {
        'goal_dim': 38,
        'proprio_dim': 13,
        'action_dim': 7,
        'chunk_size': 10,
        'hidden_dim': 128,
        'num_heads': 4,
        'num_layers': 4,
        'use_rope': True
    }

    model = AxisModel(config)
    print("Model instantiated successfully.")

    B, W, C, H, W_img = 2, 5, 3, 128, 128

    images = torch.randn(B, W, C, H, W_img)
    proprio = torch.randn(B, W, 13)
    goal = torch.randn(B, 38)
    goal_seq = torch.randn(B, W, 38)

    print("\n--- Testing Standard Path ---")
    pred_action, requery = model(images, proprio, raw_goal=goal)
    print("Action (Goal 2D):", pred_action.shape)
    print("Requery (Goal 2D):", requery.shape)
    
    pred_action, requery = model(images, proprio, raw_goal=goal_seq)
    print("Action (Goal 3D):", pred_action.shape)

    print("\n--- Testing Optimized Path ---")
    pred_action, requery, tokens = model(images, proprio, raw_goal=goal, return_tokens=True)
    cached_tokens = tokens[:, :-1, :]
    
    # Passing the exact same goal
    pred_action_opt, requery_opt, tokens_opt = model(images, proprio, raw_goal=goal, cached_tokens=cached_tokens, return_tokens=True)
    print("Action (Opt):", pred_action_opt.shape)
    print("Tokens match:", tokens.shape == tokens_opt.shape)

if __name__ == "__main__":
    main()
