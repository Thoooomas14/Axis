
from scipy.spatial.transform import Rotation as R
import numpy as np

# User values from Screenshot
# Assumed "XYZ" Euler order which is standard for most 3D software inspection panels
euler_deg = [71.08969, 0.0, -128.6459]
pos = [-0.96961, 1.03466, 0.68941]

# Create Rotation
r = R.from_euler('xyz', euler_deg, degrees=True)
quat = r.as_quat() # (x, y, z, w) in scipy

# Isaac Lab uses (w, x, y, z) usually? 
# Wait, Isaac Sim uses (w, x, y, z).
# Scipy gives (x, y, z, w).

print(f"Pos: {tuple(pos)}")
print(f"Quat (w, x, y, z): ({quat[3]:.4f}, {quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f})")
