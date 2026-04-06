import os
import sys

import h5py
import numpy as np


def write_velocity_txt(path, x, y, z, vx, vy, vz, header_lines, shuffle=False):
    rows = []
    for i, xv in enumerate(x):
        for j, yv in enumerate(y):
            for k, zv in enumerate(z):
                rows.append((xv, yv, zv, vx[i, j, k], vy[i, j, k], vz[i, j, k]))
    if shuffle:
        rng = np.random.default_rng(42)
        rng.shuffle(rows)
    with open(path, "w", encoding="utf-8") as handle:
        for line in header_lines:
            handle.write(line + "\n")
        for row in rows:
            handle.write(" ".join(f"{value:.12g}" for value in row) + "\n")


def write_scalar_txt(path, x, y, z, field, header_lines):
    with open(path, "w", encoding="utf-8") as handle:
        for line in header_lines:
            handle.write(line + "\n")
        for i, xv in enumerate(x):
            for j, yv in enumerate(y):
                for k, zv in enumerate(z):
                    handle.write(f"{xv:.12g} {yv:.12g} {zv:.12g} {field[i, j, k]:.12g}\n")


def write_velocity_h5(path, x, y, z, vx, vy, vz, periodic_duplicate_last=False):
    with h5py.File(path, "w") as hf:
        hf.attrs["schema"] = "structured_velocity_v1"
        hf.attrs["periodic_duplicate_last"] = bool(periodic_duplicate_last)
        hf.attrs["Nx"] = len(x)
        hf.attrs["Ny"] = len(y)
        hf.attrs["Nz"] = len(z)
        grid = hf.create_group("grid")
        grid.create_dataset("x", data=x)
        grid.create_dataset("y", data=y)
        grid.create_dataset("z", data=z)
        fields = hf.create_group("fields")
        fields.create_dataset("vx", data=vx)
        fields.create_dataset("vy", data=vy)
        fields.create_dataset("vz", data=vz)


def write_scalar_h5(path, x, y, z, field, periodic_duplicate_last=False):
    with h5py.File(path, "w") as hf:
        hf.attrs["schema"] = "structured_scalar_v1"
        hf.attrs["periodic_duplicate_last"] = bool(periodic_duplicate_last)
        hf.attrs["Nx"] = len(x)
        hf.attrs["Ny"] = len(y)
        hf.attrs["Nz"] = len(z)
        grid = hf.create_group("grid")
        grid.create_dataset("x", data=x)
        grid.create_dataset("y", data=y)
        grid.create_dataset("z", data=z)
        fields = hf.create_group("fields")
        fields.create_dataset("temp", data=field)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "test_input_case"
    os.makedirs(out_dir, exist_ok=True)

    n = 8
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    y = np.linspace(0.0, 1.0, n, endpoint=False)
    z = np.linspace(0.0, 1.0, n, endpoint=False)

    vx = np.ones((n, n, n), dtype=np.float64)
    vy = np.ones((n, n, n), dtype=np.float64) * 2.0
    vz = np.ones((n, n, n), dtype=np.float64) * 3.0
    scalar = np.ones((n, n, n), dtype=np.float64)

    write_velocity_txt(
        os.path.join(out_dir, "SampledDataTxt.txt"),
        x,
        y,
        z,
        vx,
        vy,
        vz,
        [
            "# Step: 0, Time: 0.0",
            "# Grid: Nx=8, Ny=8, Nz=8",
            "# Sampled velocity data",
            "",
            "# x y z vx vy vz",
        ],
        shuffle=False,
    )
    write_velocity_txt(
        os.path.join(out_dir, "SampledDataTxt_shuffled.txt"),
        x,
        y,
        z,
        vx,
        vy,
        vz,
        ["# shuffled sampled velocity", "# x y z vx vy vz"],
        shuffle=True,
    )
    write_scalar_txt(
        os.path.join(out_dir, "SampledScalar.txt"),
        x,
        y,
        z,
        scalar,
        ["# scalar sampled data", "# x y z temp", ""],
    )
    write_velocity_h5(os.path.join(out_dir, "StructuredVelocity.h5"), x, y, z, vx, vy, vz)
    write_scalar_h5(os.path.join(out_dir, "StructuredScalar.h5"), x, y, z, scalar)

    xp = np.linspace(0.0, 1.0, n + 1)
    yp = np.linspace(0.0, 1.0, n + 1)
    zp = np.linspace(0.0, 1.0, n + 1)
    vxp = np.ones((n + 1, n + 1, n + 1), dtype=np.float64)
    vyp = np.ones((n + 1, n + 1, n + 1), dtype=np.float64) * 2.0
    vzp = np.ones((n + 1, n + 1, n + 1), dtype=np.float64) * 3.0
    scalarp = np.ones((n + 1, n + 1, n + 1), dtype=np.float64)

    write_velocity_txt(
        os.path.join(out_dir, "PeriodicSampledDataTxt.txt"),
        xp,
        yp,
        zp,
        vxp,
        vyp,
        vzp,
        ["# periodic sampled velocity", "# x y z vx vy vz"],
        shuffle=False,
    )
    write_scalar_txt(
        os.path.join(out_dir, "PeriodicSampledScalar.txt"),
        xp,
        yp,
        zp,
        scalarp,
        ["# periodic sampled scalar", "# x y z temp"],
    )
    write_velocity_h5(os.path.join(out_dir, "PeriodicStructuredVelocity.h5"), xp, yp, zp, vxp, vyp, vzp, periodic_duplicate_last=True)
    write_scalar_h5(os.path.join(out_dir, "PeriodicStructuredScalar.h5"), xp, yp, zp, scalarp, periodic_duplicate_last=True)


if __name__ == "__main__":
    main()
