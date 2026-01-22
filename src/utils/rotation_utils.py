"""
Rotation utility functions for SE(3) Lie group operations.

Uses PyPose library for differentiable Lie group operations:
- pypose.SE3, pypose.SO3 for Lie group types
- pypose.Exp, pypose.Log for exponential/log maps

Also includes:
- 6D continuous rotation representation (Zhou et al., 2019)
- Chordal loss for SE(3) comparison
"""

import torch
import pypose as pp


# =============================================================================
# 6D Continuous Rotation Representation (Zhou et al., 2019)
# =============================================================================

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    
    Uses Gram-Schmidt orthonormalization - NO trigonometric functions.
    
    Args:
        d6: (..., 6) tensor - first two columns of rotation matrix flattened
        
    Returns:
        (..., 3, 3) rotation matrix
    """
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    
    # Gram-Schmidt orthonormalization
    b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
    dot = torch.sum(b1 * a2, dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to 6D rotation representation.
    
    Args:
        matrix: (..., 3, 3) rotation matrix
        
    Returns:
        (..., 6) tensor - first two columns of rotation matrix flattened
    """
    col1 = matrix[..., :, 0]
    col2 = matrix[..., :, 1]
    return torch.cat([col1, col2], dim=-1)


# =============================================================================
# Quaternion <-> Matrix Conversions (for data loading)
# =============================================================================

def quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 3x3 rotation matrix.
    
    Args:
        quat: (..., 4) quaternion in [x, y, z, w] format
        
    Returns:
        (..., 3, 3) rotation matrix
    """
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    
    matrix = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)
    
    return matrix


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to quaternion using Shepperd's method.
    
    Args:
        matrix: (..., 3, 3) rotation matrix
        
    Returns:
        (..., 4) quaternion in [x, y, z, w] format
    """
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)
    n = m.shape[0]
    
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    
    trace = m00 + m11 + m22
    quat = torch.zeros(n, 4, device=matrix.device, dtype=matrix.dtype)
    
    # Case 1: trace > 0
    mask = trace > 0
    if mask.any():
        s = torch.sqrt(trace[mask] + 1.0) * 2
        quat[mask, 3] = 0.25 * s
        quat[mask, 0] = (m21[mask] - m12[mask]) / s
        quat[mask, 1] = (m02[mask] - m20[mask]) / s
        quat[mask, 2] = (m10[mask] - m01[mask]) / s
    
    # Case 2: m00 largest
    mask = (~(trace > 0)) & (m00 > m11) & (m00 > m22)
    if mask.any():
        s = torch.sqrt(1.0 + m00[mask] - m11[mask] - m22[mask]) * 2
        quat[mask, 3] = (m21[mask] - m12[mask]) / s
        quat[mask, 0] = 0.25 * s
        quat[mask, 1] = (m01[mask] + m10[mask]) / s
        quat[mask, 2] = (m02[mask] + m20[mask]) / s
    
    # Case 3: m11 largest
    mask = (~(trace > 0)) & (~((m00 > m11) & (m00 > m22))) & (m11 > m22)
    if mask.any():
        s = torch.sqrt(1.0 + m11[mask] - m00[mask] - m22[mask]) * 2
        quat[mask, 3] = (m02[mask] - m20[mask]) / s
        quat[mask, 0] = (m01[mask] + m10[mask]) / s
        quat[mask, 1] = 0.25 * s
        quat[mask, 2] = (m12[mask] + m21[mask]) / s
    
    # Case 4: m22 largest
    mask = (~(trace > 0)) & (~((m00 > m11) & (m00 > m22))) & (~(m11 > m22))
    if mask.any():
        s = torch.sqrt(1.0 + m22[mask] - m00[mask] - m11[mask]) * 2
        quat[mask, 3] = (m10[mask] - m01[mask]) / s
        quat[mask, 0] = (m02[mask] + m20[mask]) / s
        quat[mask, 1] = (m12[mask] + m21[mask]) / s
        quat[mask, 2] = 0.25 * s
    
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
    return quat.reshape(*batch_shape, 4)


def quaternion_to_rotation_6d(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to 6D rotation representation."""
    return matrix_to_rotation_6d(quaternion_to_matrix(quat))


def rotation_6d_to_quaternion(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to quaternion."""
    return matrix_to_quaternion(rotation_6d_to_matrix(d6))


# =============================================================================
# 13D Pose Representation: [R_flat(9), pos(3), gripper(1)]
# =============================================================================

def pose_13d_to_se3_matrix(pose_13d: torch.Tensor) -> torch.Tensor:
    """
    Convert 13D pose [R_flat(9), pos(3), gripper(1)] to 4x4 SE(3) matrix.
    
    Args:
        pose_13d: (..., 13) pose vector
        
    Returns:
        (..., 4, 4) SE(3) transformation matrix
    """
    batch_shape = pose_13d.shape[:-1]
    R_flat = pose_13d[..., :9]
    pos = pose_13d[..., 9:12]
    
    # Reshape R_flat to 3x3 matrix (column-major order)
    R = R_flat.reshape(*batch_shape, 3, 3)
    
    # Build 4x4 matrix
    T = torch.eye(4, device=pose_13d.device, dtype=pose_13d.dtype)
    if len(batch_shape) > 0:
        T = T.unsqueeze(0).expand(*batch_shape, 4, 4).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = pos
    
    return T


def se3_matrix_to_pose_13d(T: torch.Tensor, gripper: torch.Tensor = None) -> torch.Tensor:
    """
    Convert 4x4 SE(3) matrix to 13D pose [R_flat(9), pos(3), gripper(1)].
    
    Args:
        T: (..., 4, 4) SE(3) transformation matrix
        gripper: (..., 1) or (...,) gripper state (if None, uses 0.0)
        
    Returns:
        (..., 13) pose vector
    """
    batch_shape = T.shape[:-2]
    R = T[..., :3, :3]
    pos = T[..., :3, 3]
    
    R_flat = R.reshape(*batch_shape, 9)
    
    if gripper is None:
        gripper = torch.zeros(*batch_shape, 1, device=T.device, dtype=T.dtype)
    elif gripper.dim() == len(batch_shape):
        gripper = gripper.unsqueeze(-1)
    
    return torch.cat([R_flat, pos, gripper], dim=-1)


# =============================================================================
# PyPose SE(3) Operations
# =============================================================================

def apply_twist(T: torch.Tensor, twist_6d: torch.Tensor) -> torch.Tensor:
    """
    Apply a 6D twist to an SE(3) transformation using PyPose.
    
    Args:
        T: (B, 4, 4) current SE(3) transformation matrix
        twist_6d: (B, 6) twist vector [ω_x, ω_y, ω_z, v_x, v_y, v_z]
        
    Returns:
        (B, 4, 4) new SE(3) transformation matrix
    """
    # Convert twist to se3 Lie algebra element and apply exp map
    delta_se3 = pp.se3(twist_6d)  # LieTensor in se(3)
    delta_SE3 = pp.Exp(delta_se3)  # LieTensor in SE(3)
    
    # Convert current T to PyPose SE3
    T_pp = pp.mat2SE3(T)
    
    # Compose: T_new = T @ delta (body-frame twist)
    T_new_pp = T_pp @ delta_SE3
    
    # Convert back to matrix
    return T_new_pp.matrix()


def compute_twist(T_curr: torch.Tensor, T_next: torch.Tensor) -> torch.Tensor:
    """
    Compute the twist that transforms T_curr to T_next.
    
    ξ = log(T_curr⁻¹ @ T_next)
    
    Args:
        T_curr: (B, 4, 4) current SE(3) matrix
        T_next: (B, 4, 4) next SE(3) matrix
        
    Returns:
        (B, 6) twist vector [ω_x, ω_y, ω_z, v_x, v_y, v_z]
    """
    T_curr_pp = pp.mat2SE3(T_curr)
    T_next_pp = pp.mat2SE3(T_next)
    
    # T_rel = T_curr^(-1) @ T_next
    T_rel_pp = T_curr_pp.Inv() @ T_next_pp
    
    # Log map to get twist in Lie algebra
    twist = pp.Log(T_rel_pp)
    
    return twist.tensor()


# =============================================================================
# Chordal Loss (Trig-Free!)
# =============================================================================

def chordal_rotation_loss(R_pred: torch.Tensor, R_target: torch.Tensor) -> torch.Tensor:
    """
    Compute chordal distance on SO(3): ||R_pred - R_target||²_F
    
    This is the squared Frobenius norm of the rotation matrix difference.
    No trigonometric functions!
    
    Args:
        R_pred: (..., 3, 3) predicted rotation matrix
        R_target: (..., 3, 3) target rotation matrix
        
    Returns:
        Scalar loss (mean over batch)
    """
    diff = R_pred - R_target
    # Frobenius norm squared
    frob_sq = torch.sum(diff ** 2, dim=(-2, -1))
    return frob_sq.mean()


def chordal_translation_loss(p_pred: torch.Tensor, p_target: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 loss on translation: ||p_pred - p_target||²
    
    Args:
        p_pred: (..., 3) predicted translation
        p_target: (..., 3) target translation
        
    Returns:
        Scalar loss (mean over batch)
    """
    diff_sq = torch.sum((p_pred - p_target) ** 2, dim=-1)
    return diff_sq.mean()


def chordal_se3_loss(
    T_pred: torch.Tensor,
    T_target: torch.Tensor,
    omega_rot: float = 1.0,
    omega_trans: float = 1.0
) -> torch.Tensor:
    """
    Compute weighted chordal loss on SE(3).
    
    L = ω_rot * ||R_pred - R_target||²_F + ω_trans * ||p_pred - p_target||²
    
    No trigonometric functions - direct matrix comparison!
    
    Args:
        T_pred: (..., 4, 4) predicted SE(3) matrix
        T_target: (..., 4, 4) target SE(3) matrix
        omega_rot: weight for rotation component
        omega_trans: weight for translation component
        
    Returns:
        Scalar loss
    """
    R_pred = T_pred[..., :3, :3]
    R_target = T_target[..., :3, :3]
    p_pred = T_pred[..., :3, 3]
    p_target = T_target[..., :3, 3]
    
    rot_loss = chordal_rotation_loss(R_pred, R_target)
    trans_loss = chordal_translation_loss(p_pred, p_target)
    
    return omega_rot * rot_loss + omega_trans * trans_loss


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing rotation utilities with PyPose...")
    
    # Test 6D representation roundtrip
    quat = torch.tensor([0.0, 0.0, 0.7071, 0.7071])  # 90° around Z
    d6 = quaternion_to_rotation_6d(quat)
    quat_back = rotation_6d_to_quaternion(d6)
    print(f"6D roundtrip: {quat} -> {quat_back}")
    assert torch.allclose(quat.abs(), quat_back.abs(), atol=1e-5)
    
    # Test 13D pose conversion
    pose_13d = torch.randn(2, 13)
    T = pose_13d_to_se3_matrix(pose_13d)
    print(f"13D -> SE(3): {pose_13d.shape} -> {T.shape}")
    assert T.shape == (2, 4, 4)
    
    # Test chordal loss
    T1 = torch.eye(4).unsqueeze(0)
    T2 = torch.eye(4).unsqueeze(0)
    T2[0, 0, 3] = 0.1  # Small translation
    loss = chordal_se3_loss(T1, T2)
    print(f"Chordal loss (small diff): {loss.item():.6f}")
    assert loss.item() > 0
    
    # Test PyPose twist operations
    T_curr = torch.eye(4).unsqueeze(0)
    twist = torch.tensor([[0.0, 0.0, 0.1, 0.01, 0.0, 0.0]])  # Small rotation + translation
    T_new = apply_twist(T_curr, twist)
    print(f"Apply twist: {T_curr.shape} + {twist.shape} -> {T_new.shape}")
    
    # Verify roundtrip
    twist_back = compute_twist(T_curr, T_new)
    print(f"Twist roundtrip: {twist} -> {twist_back}")
    assert torch.allclose(twist, twist_back, atol=1e-5)
    
    print("All tests passed!")
