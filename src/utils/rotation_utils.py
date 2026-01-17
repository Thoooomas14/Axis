"""
Rotation utility functions for converting between rotation representations.

Implements conversions between:
- 6D continuous rotation representation (Zhou et al., 2019)
- 3x3 rotation matrices
- Quaternions [x, y, z, w]

The 6D representation uses the first two columns of a rotation matrix,
providing better gradient flow for neural networks (no discontinuities).

Reference: "On the Continuity of Rotation Representations in Neural Networks"
           Zhou et al., CVPR 2019
"""

import torch


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    
    Uses Gram-Schmidt orthonormalization to ensure valid rotation matrix.
    
    Args:
        d6: (..., 6) tensor - first two columns of rotation matrix flattened
        
    Returns:
        (..., 3, 3) rotation matrix
    """
    # Reshape to get the two column vectors
    a1 = d6[..., :3]  # (..., 3) - first column
    a2 = d6[..., 3:]  # (..., 3) - second column
    
    # Gram-Schmidt orthonormalization
    # Normalize first vector
    b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
    
    # Make second vector orthogonal to first and normalize
    dot = torch.sum(b1 * a2, dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
    
    # Third column is cross product of first two
    b3 = torch.cross(b1, b2, dim=-1)
    
    # Stack columns into rotation matrix
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to 6D rotation representation.
    
    Args:
        matrix: (..., 3, 3) rotation matrix
        
    Returns:
        (..., 6) tensor - first two columns of rotation matrix flattened
    """
    # Extract first two columns: (..., 3, 2) then transpose to (..., 2, 3) and flatten
    # Column 1 = matrix[..., :, 0], Column 2 = matrix[..., :, 1]
    batch_shape = matrix.shape[:-2]
    col1 = matrix[..., :, 0]  # (..., 3)
    col2 = matrix[..., :, 1]  # (..., 3)
    return torch.cat([col1, col2], dim=-1)  # (..., 6)


def quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 3x3 rotation matrix.
    
    Args:
        quat: (..., 4) quaternion in [x, y, z, w] format
        
    Returns:
        (..., 3, 3) rotation matrix
    """
    # Normalize quaternion
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
    
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # Rotation matrix from quaternion
    # First row
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    
    # Second row
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    
    # Third row
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    
    # Stack into matrix
    matrix = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)
    
    return matrix


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to quaternion.
    
    Uses Shepperd's method for numerical stability.
    
    Args:
        matrix: (..., 3, 3) rotation matrix
        
    Returns:
        (..., 4) quaternion in [x, y, z, w] format
    """
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)
    batch_size = m.shape[0]
    
    # Shepperd's method - choose the largest diagonal element for stability
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    
    trace = m00 + m11 + m22
    
    # Initialize output
    quat = torch.zeros(batch_size, 4, device=matrix.device, dtype=matrix.dtype)
    
    # Case 1: trace > 0
    mask = trace > 0
    if mask.any():
        s = torch.sqrt(trace[mask] + 1.0) * 2  # s = 4 * w
        quat[mask, 3] = 0.25 * s
        quat[mask, 0] = (m21[mask] - m12[mask]) / s
        quat[mask, 1] = (m02[mask] - m20[mask]) / s
        quat[mask, 2] = (m10[mask] - m01[mask]) / s
    
    # Case 2: m00 is largest diagonal
    mask = (~(trace > 0)) & (m00 > m11) & (m00 > m22)
    if mask.any():
        s = torch.sqrt(1.0 + m00[mask] - m11[mask] - m22[mask]) * 2  # s = 4 * x
        quat[mask, 3] = (m21[mask] - m12[mask]) / s
        quat[mask, 0] = 0.25 * s
        quat[mask, 1] = (m01[mask] + m10[mask]) / s
        quat[mask, 2] = (m02[mask] + m20[mask]) / s
    
    # Case 3: m11 is largest diagonal
    mask = (~(trace > 0)) & (~((m00 > m11) & (m00 > m22))) & (m11 > m22)
    if mask.any():
        s = torch.sqrt(1.0 + m11[mask] - m00[mask] - m22[mask]) * 2  # s = 4 * y
        quat[mask, 3] = (m02[mask] - m20[mask]) / s
        quat[mask, 0] = (m01[mask] + m10[mask]) / s
        quat[mask, 1] = 0.25 * s
        quat[mask, 2] = (m12[mask] + m21[mask]) / s
    
    # Case 4: m22 is largest diagonal
    mask = (~(trace > 0)) & (~((m00 > m11) & (m00 > m22))) & (~(m11 > m22))
    if mask.any():
        s = torch.sqrt(1.0 + m22[mask] - m00[mask] - m11[mask]) * 2  # s = 4 * z
        quat[mask, 3] = (m10[mask] - m01[mask]) / s
        quat[mask, 0] = (m02[mask] + m20[mask]) / s
        quat[mask, 1] = (m12[mask] + m21[mask]) / s
        quat[mask, 2] = 0.25 * s
    
    # Normalize quaternion
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
    
    # Reshape back to original batch shape
    return quat.reshape(*batch_shape, 4)


def quaternion_to_rotation_6d(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 6D rotation representation.
    
    Args:
        quat: (..., 4) quaternion in [x, y, z, w] format
        
    Returns:
        (..., 6) tensor - 6D rotation representation
    """
    matrix = quaternion_to_matrix(quat)
    return matrix_to_rotation_6d(matrix)


def rotation_6d_to_quaternion(d6: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to quaternion.
    
    Args:
        d6: (..., 6) tensor - 6D rotation representation
        
    Returns:
        (..., 4) quaternion in [x, y, z, w] format
    """
    matrix = rotation_6d_to_matrix(d6)
    return matrix_to_quaternion(matrix)


# =============================================================================
# SE(3) Lie Group Utilities
# =============================================================================

def so3_log(R: torch.Tensor) -> torch.Tensor:
    """
    Compute the logarithmic map of SO(3): rotation matrix -> rotation vector.
    
    The rotation vector (axis-angle) has magnitude = angle and direction = axis.
    
    Args:
        R: (..., 3, 3) rotation matrix
        
    Returns:
        (..., 3) rotation vector (axis-angle representation)
    """
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    n = R_flat.shape[0]
    
    # Compute rotation angle from trace: trace(R) = 1 + 2*cos(theta)
    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    cos_angle = (trace - 1) / 2  # Avoid float literal, use integer
    cos_angle = torch.clamp(cos_angle, -1 + 1e-7, 1 - 1e-7)  # Numerical stability
    angle = torch.acos(cos_angle)  # (n,)
    
    # Initialize output
    omega = torch.zeros(n, 3, device=R.device, dtype=R.dtype)
    
    # Case 1: Small angle (theta ≈ 0) - use first-order approximation
    small_mask = angle.abs() < 1e-6
    if small_mask.any():
        # For small angles: omega ≈ [R32-R23, R13-R31, R21-R12] / 2
        # Use integer 2 to preserve dtype
        omega[small_mask, 0] = (R_flat[small_mask, 2, 1] - R_flat[small_mask, 1, 2]) / 2
        omega[small_mask, 1] = (R_flat[small_mask, 0, 2] - R_flat[small_mask, 2, 0]) / 2
        omega[small_mask, 2] = (R_flat[small_mask, 1, 0] - R_flat[small_mask, 0, 1]) / 2
    
    # Case 2: Normal case
    normal_mask = ~small_mask & (angle < (3.14159 - 1e-6))
    if normal_mask.any():
        # omega = (theta / (2*sin(theta))) * [R32-R23, R13-R31, R21-R12]
        sin_angle = torch.sin(angle[normal_mask])
        factor = angle[normal_mask] / (2 * sin_angle + 1e-8)
        omega[normal_mask, 0] = factor * (R_flat[normal_mask, 2, 1] - R_flat[normal_mask, 1, 2])
        omega[normal_mask, 1] = factor * (R_flat[normal_mask, 0, 2] - R_flat[normal_mask, 2, 0])
        omega[normal_mask, 2] = factor * (R_flat[normal_mask, 1, 0] - R_flat[normal_mask, 0, 1])
    
    # Case 3: angle ≈ π (singularity) - use diagonal elements
    large_mask = angle >= (3.14159 - 1e-6)
    if large_mask.any():
        # Find largest diagonal element to determine axis
        diag = torch.stack([R_flat[large_mask, 0, 0], 
                           R_flat[large_mask, 1, 1], 
                           R_flat[large_mask, 2, 2]], dim=-1)
        k = torch.argmax(diag, dim=-1)
        
        for idx in range(large_mask.sum()):
            i = k[idx]
            j = (i + 1) % 3
            l = (i + 2) % 3
            R_i = R_flat[large_mask][idx]
            s = torch.sqrt(R_i[i, i] - R_i[j, j] - R_i[l, l] + 1) + 1e-8
            w = torch.zeros(3, device=R.device, dtype=R.dtype)
            w[i] = s / 2
            w[j] = (R_i[i, j] + R_i[j, i]) / (2 * s)
            w[l] = (R_i[i, l] + R_i[l, i]) / (2 * s)
            omega[large_mask][idx] = w * angle[large_mask][idx]
    
    return omega.reshape(*batch_shape, 3)


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """
    Compute the exponential map of SO(3): rotation vector -> rotation matrix.
    
    Uses Rodrigues' formula: R = I + sin(θ)K + (1 - cos(θ))K²
    where K is the skew-symmetric matrix of the unit axis.
    
    Args:
        omega: (..., 3) rotation vector (axis-angle)
        
    Returns:
        (..., 3, 3) rotation matrix
    """
    batch_shape = omega.shape[:-1]
    omega_flat = omega.reshape(-1, 3)
    n = omega_flat.shape[0]
    
    # Compute angle
    angle = torch.norm(omega_flat, dim=-1, keepdim=True)  # (n, 1)
    
    # Handle small angles with Taylor expansion
    small_mask = (angle < 1e-6).squeeze(-1)
    
    # Initialize with identity
    R = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0).expand(n, 3, 3).clone()
    
    # Normal case: use Rodrigues' formula
    normal_mask = ~small_mask
    if normal_mask.any():
        axis = omega_flat[normal_mask] / (angle[normal_mask] + 1e-8)  # Unit axis
        theta = angle[normal_mask, 0]  # (m,)
        
        # Skew symmetric matrix K
        K = torch.zeros(normal_mask.sum(), 3, 3, device=omega.device, dtype=omega.dtype)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        # Rodrigues: R = I + sin(θ)K + (1-cos(θ))K²
        sin_t = torch.sin(theta).unsqueeze(-1).unsqueeze(-1)
        cos_t = torch.cos(theta).unsqueeze(-1).unsqueeze(-1)
        I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)
        K2 = torch.bmm(K, K)
        # Use (1 - cos_t) instead of (1.0 - cos_t) to preserve dtype
        one = torch.ones_like(cos_t)
        R[normal_mask] = I + sin_t * K + (one - cos_t) * K2
    
    # Small angle: R ≈ I + K (first order Taylor)
    if small_mask.any():
        K_small = torch.zeros(small_mask.sum(), 3, 3, device=omega.device, dtype=omega.dtype)
        w = omega_flat[small_mask]
        K_small[:, 0, 1] = -w[:, 2]
        K_small[:, 0, 2] = w[:, 1]
        K_small[:, 1, 0] = w[:, 2]
        K_small[:, 1, 2] = -w[:, 0]
        K_small[:, 2, 0] = -w[:, 1]
        K_small[:, 2, 1] = w[:, 0]
        I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)
        R[small_mask] = I + K_small
    
    return R.reshape(*batch_shape, 3, 3)


def se3_log(T: torch.Tensor) -> torch.Tensor:
    """
    Compute the logarithmic map of SE(3): transformation matrix -> 6D twist.
    
    The twist ξ = [ω, v] where ω is the rotational component and v is translational.
    
    Args:
        T: (..., 4, 4) SE(3) transformation matrix
        
    Returns:
        (..., 6) twist vector [ω_x, ω_y, ω_z, v_x, v_y, v_z]
    """
    batch_shape = T.shape[:-2]
    T_flat = T.reshape(-1, 4, 4)
    n = T_flat.shape[0]
    
    # Extract rotation and translation
    R = T_flat[:, :3, :3]  # (n, 3, 3)
    p = T_flat[:, :3, 3]   # (n, 3)
    
    # Compute rotation vector
    omega = so3_log(R)  # (n, 3)
    theta = torch.norm(omega, dim=-1, keepdim=True)  # (n, 1)
    
    # Compute V^{-1} to get v from p: p = V*v, so v = V^{-1}*p
    # V = I + (1-cos(θ))/θ² * K + (θ - sin(θ))/θ³ * K²
    # V^{-1} = I - K/2 + (1/θ² - (1+cos(θ))/(2θ*sin(θ))) * K²
    
    v = torch.zeros(n, 3, device=T.device, dtype=T.dtype)
    
    # Small angle case: V ≈ I, so v ≈ p
    small_mask = (theta < 1e-6).squeeze(-1)
    v[small_mask] = p[small_mask]
    
    # Normal case
    normal_mask = ~small_mask
    if normal_mask.any():
        omega_n = omega[normal_mask]
        theta_n = theta[normal_mask, 0]
        p_n = p[normal_mask]
        
        # Unit axis
        axis = omega_n / (theta_n.unsqueeze(-1) + 1e-8)
        
        # Build skew-symmetric K
        K = torch.zeros(normal_mask.sum(), 3, 3, device=T.device, dtype=T.dtype)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        K2 = torch.bmm(K, K)
        
        # V^{-1} coefficients
        sin_t = torch.sin(theta_n)
        cos_t = torch.cos(theta_n)
        
        # Coefficient for K²: (1/θ² - (1+cos(θ))/(2θ*sin(θ)))
        # Use tensor operations to preserve dtype for AMP
        one = torch.ones_like(cos_t)
        two = one + one
        c2 = (one / (theta_n**2 + 1e-8) - (one + cos_t) / (two * theta_n * sin_t + 1e-8))
        
        I = torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0)
        half = one * 0.5  # Preserve dtype
        V_inv = I - half.unsqueeze(-1).unsqueeze(-1) * K + c2.unsqueeze(-1).unsqueeze(-1) * K2
        
        v[normal_mask] = torch.bmm(V_inv, p_n.unsqueeze(-1)).squeeze(-1)
    
    # Combine to twist
    xi = torch.cat([omega, v], dim=-1)  # (n, 6)
    return xi.reshape(*batch_shape, 6)


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """
    Compute the exponential map of SE(3): 6D twist -> transformation matrix.
    
    Args:
        xi: (..., 6) twist vector [ω_x, ω_y, ω_z, v_x, v_y, v_z]
        
    Returns:
        (..., 4, 4) SE(3) transformation matrix
    """
    batch_shape = xi.shape[:-1]
    xi_flat = xi.reshape(-1, 6)
    n = xi_flat.shape[0]
    
    omega = xi_flat[:, :3]  # (n, 3)
    v = xi_flat[:, 3:]      # (n, 3)
    
    # Compute rotation matrix
    R = so3_exp(omega)  # (n, 3, 3)
    
    # Compute translation: p = V * v
    theta = torch.norm(omega, dim=-1, keepdim=True)  # (n, 1)
    
    p = torch.zeros(n, 3, device=xi.device, dtype=xi.dtype)
    
    # Small angle: V ≈ I
    small_mask = (theta < 1e-6).squeeze(-1)
    p[small_mask] = v[small_mask]
    
    # Normal case: p = V * v
    normal_mask = ~small_mask
    if normal_mask.any():
        omega_n = omega[normal_mask]
        theta_n = theta[normal_mask, 0]
        v_n = v[normal_mask]
        
        axis = omega_n / (theta_n.unsqueeze(-1) + 1e-8)
        
        # Build K
        K = torch.zeros(normal_mask.sum(), 3, 3, device=xi.device, dtype=xi.dtype)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        K2 = torch.bmm(K, K)
        
        sin_t = torch.sin(theta_n)
        cos_t = torch.cos(theta_n)
        
        # V = I + (1-cos)/θ² * K + (θ-sin)/θ³ * K²
        # Use tensor operations to preserve dtype for AMP
        one = torch.ones_like(cos_t)
        c1 = ((one - cos_t) / (theta_n**2 + 1e-8)).unsqueeze(-1).unsqueeze(-1)
        c2 = ((theta_n - sin_t) / (theta_n**3 + 1e-8)).unsqueeze(-1).unsqueeze(-1)
        
        I = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0)
        V = I + c1 * K + c2 * K2
        
        p[normal_mask] = torch.bmm(V, v_n.unsqueeze(-1)).squeeze(-1)
    
    # Construct SE(3) matrix
    T = torch.eye(4, device=xi.device, dtype=xi.dtype).unsqueeze(0).expand(n, 4, 4).clone()
    T[:, :3, :3] = R
    T[:, :3, 3] = p
    
    return T.reshape(*batch_shape, 4, 4)


def pose_to_se3(pose: torch.Tensor, format: str = 'quat') -> torch.Tensor:
    """
    Convert pose vector to SE(3) transformation matrix.
    
    Args:
        pose: Pose vector. For 'quat': (..., 7) [x, y, z, qx, qy, qz, qw]
              For 'rotvec': (..., 6) [φ_x, φ_y, φ_z, p_x, p_y, p_z]
        format: 'quat' or 'rotvec'
        
    Returns:
        (..., 4, 4) SE(3) transformation matrix
    """
    batch_shape = pose.shape[:-1]
    
    if format == 'quat':
        # pose: [x, y, z, qx, qy, qz, qw] (7D)
        pos = pose[..., :3]
        quat = pose[..., 3:7]
        R = quaternion_to_matrix(quat)
    elif format == 'rotvec':
        # pose: [φ_x, φ_y, φ_z, p_x, p_y, p_z] (6D)
        omega = pose[..., :3]
        pos = pose[..., 3:6]
        R = so3_exp(omega)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    # Construct SE(3)
    T = torch.eye(4, device=pose.device, dtype=pose.dtype)
    if len(batch_shape) > 0:
        T = T.unsqueeze(0).expand(*batch_shape, 4, 4).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = pos
    
    return T


if __name__ == "__main__":
    # Test roundtrip: quat -> 6d -> quat
    quat = torch.tensor([0.0, 0.0, 0.7071, 0.7071])  # 90° around Z
    d6 = quaternion_to_rotation_6d(quat)
    quat_back = rotation_6d_to_quaternion(d6)
    print(f"Original: {quat}, Recovered: {quat_back}")
    assert torch.allclose(quat.abs(), quat_back.abs(), atol=1e-5)
    
    # Additional tests
    print("\n--- Additional Tests ---")
    
    # Test identity quaternion
    identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    identity_d6 = quaternion_to_rotation_6d(identity_quat)
    identity_matrix = rotation_6d_to_matrix(identity_d6)
    print(f"Identity matrix:\n{identity_matrix}")
    assert torch.allclose(identity_matrix, torch.eye(3), atol=1e-5)
    
    # Test batch processing
    batch_quat = torch.tensor([
        [0.0, 0.0, 0.0, 1.0],      # identity
        [0.0, 0.0, 0.7071, 0.7071], # 90° around Z
        [0.7071, 0.0, 0.0, 0.7071], # 90° around X
    ])
    batch_d6 = quaternion_to_rotation_6d(batch_quat)
    batch_quat_back = rotation_6d_to_quaternion(batch_d6)
    print(f"\nBatch test - Original shape: {batch_quat.shape}, Recovered shape: {batch_quat_back.shape}")
    assert torch.allclose(batch_quat.abs(), batch_quat_back.abs(), atol=1e-5)
    
    # Test matrix -> 6d -> matrix roundtrip
    random_matrix = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0]
    ])  # 90° rotation around X
    d6_from_mat = matrix_to_rotation_6d(random_matrix)
    matrix_back = rotation_6d_to_matrix(d6_from_mat)
    print(f"\nMatrix roundtrip test:")
    print(f"Original:\n{random_matrix}")
    print(f"Recovered:\n{matrix_back}")
    assert torch.allclose(random_matrix, matrix_back, atol=1e-5)
    
    print("\n✓ All tests passed!")
