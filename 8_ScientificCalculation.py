import numpy as np

# scalar example
print("sin(pi/2) =", np.sin(np.pi / 2))

# simple vector example (degrees -> radians)
angles_deg = [0, 30, 90]
angles_rad = np.deg2rad(angles_deg)
print("angles (deg):", angles_deg)
print("sin(angles):", np.sin(angles_rad))