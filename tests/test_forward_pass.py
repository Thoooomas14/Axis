import pytest
import torch
from src.models.axis import AxisModel

@pytest.fixture
def model_and_data():
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
    model.eval() # MUST be in eval mode to prevent BatchNorm tracking batch-size dependent stats
    
    B, W, C, H, W_img = 2, 5, 3, 128, 128
    
    images = torch.randn(B, W, C, H, W_img)
    proprio = torch.randn(B, W, 13)
    goal = torch.randn(B, 38)
    goal_seq = torch.randn(B, W, 38)
    
    return model, B, images, proprio, goal, goal_seq


def test_standard_path_2d_goal(model_and_data):
    model, B, images, proprio, goal, _ = model_and_data
    pred_action, requery = model(images, proprio, raw_goal=goal)
    
    assert pred_action.shape == (B, 10, 7)
    assert requery.shape == (B, 1)


def test_standard_path_3d_goal(model_and_data):
    model, B, images, proprio, _, goal_seq = model_and_data
    pred_action, requery = model(images, proprio, raw_goal=goal_seq)
    
    assert pred_action.shape == (B, 10, 7)
    assert requery.shape == (B, 1)


def test_optimized_path(model_and_data):
    model, B, images, proprio, goal, _ = model_and_data
    
    # 1. Full un-cached pass
    pred_action, requery, tokens = model(
        images, proprio, raw_goal=goal, return_tokens=True
    )
    cached_tokens = tokens[:, :-1, :]
    
    # 2. Optimized pass (using cached tokens up to W-1)
    pred_action_opt, requery_opt, tokens_opt = model(
        images, proprio, raw_goal=goal, 
        cached_tokens=cached_tokens, return_tokens=True
    )
    
    assert pred_action_opt.shape == (B, 10, 7)
    assert tokens.shape == tokens_opt.shape
    
    # Using torch.allclose to assert the cached path computes identical tokens.
    # We relax the tolerance to 1e-4 because optimized path involves slightly different
    # matmuls (1 frame vs 5 frames) which accumulate minor float32 discrepancies.
    assert torch.allclose(tokens, tokens_opt, atol=1e-4)
