import h5py
import numpy as np
import os
import sys

out_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
os.makedirs(out_dir, exist_ok=True)

N = 8
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
z = np.linspace(0, 1, N)

vx = np.ones((N, N, N))
vy = np.ones((N, N, N)) * 2
vz = np.ones((N, N, N)) * 3

# TKE = 0.5 * (1^2 + 2^2 + 3^2) * 8^3 = 0.5 * (1+4+9) * 512 = 7 * 512 = 3584

with h5py.File(os.path.join(out_dir, 'TPP_Small.h5'), 'w') as hf:
    hf.attrs['schema'] = 'structured_velocity_v1'
    grid = hf.create_group('grid')
    grid.create_dataset('x', data=x)
    grid.create_dataset('y', data=y)
    grid.create_dataset('z', data=z)
    fields = hf.create_group('fields')
    fields.create_dataset('vx', data=vx)
    fields.create_dataset('vy', data=vy)
    fields.create_dataset('vz', data=vz)

with h5py.File(os.path.join(out_dir, 'TPP_Small_Scalar.h5'), 'w') as hf:
    hf.attrs['schema'] = 'structured_scalar_v1'
    grid = hf.create_group('grid')
    grid.create_dataset('x', data=x)
    grid.create_dataset('y', data=y)
    grid.create_dataset('z', data=z)
    fields = hf.create_group('fields')
    fields.create_dataset('temp', data=vx) # SumSq = 1^2 * 512 = 512
