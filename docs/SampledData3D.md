# 3D Sampled-Data Workflow

This note documents the recommended `fastSF` workflow for 3D periodic sampled-data inputs generated in the same style as `turbulence_post_process`.

## Supported Input Modes

`fastSF` now supports two direct velocity input paths for 3D sampled data:

1. Raw sampled-data text files with rows of:

```text
x y z vx vy vz
```

2. Structured HDF5 files with the schema:

```text
/grid/x
/grid/y
/grid/z
/fields/vx
/fields/vy
/fields/vz
```

For scalar data, the corresponding structured HDF5 layout is:

```text
/grid/x
/grid/y
/grid/z
/fields/<scalar_name>
```

## Runtime Behavior

When a `.txt` input is provided:

1. `fastSF` detects the header length automatically.
2. The sampled-data text file is converted to structured HDF5 in parallel using MPI.
3. If the grid contains the duplicated periodic endpoint, the last plane in each periodic direction is removed before writing the structured HDF5 file.
4. The converted HDF5 file is verified.
5. The original `.txt` file is deleted only after successful verification.
6. The run prints `TOTAL KINETIC ENERGY (or Scalar SumSq): ...`.
7. The structure functions are computed from the converted HDF5 file.

When an `.h5` input is provided:

1. The text conversion step is skipped.
2. If the HDF5 file advertises `periodic_duplicate_last=true`, `fastSF` trims the duplicated endpoint planes on read.
3. The run prints `TOTAL KINETIC ENERGY (or Scalar SumSq): ...`.

If you want to preserve the original sampled-data text file, run on a copy because successful conversion removes the source `.txt`.

## Recommended `para.yaml`

For 3D periodic velocity runs, start with:

```yaml
program:
    scalar_switch: false
    2D_switch : false
    Only_longitudinal: true
    Processors_X: 2

domain_dimension :
    Lx : 1.0
    Ly : 1.0
    Lz : 1.0

structure_function :
    q1 : 1
    q2 : 1

test :
    test_switch : false
```

Notes:

- `scalar_switch: false` is for velocity data.
- `2D_switch : false` is required for 3D runs.
- `Only_longitudinal: true` is a good production default when you want the cheapest run first.
- A `grid:` block is not needed for sampled-data input. The grid is inferred from the data file.

## Recommended Commands

If the sampled-data file is in `data/`, run:

```bash
mpirun -np 4 src/fastSF.out -U data/SampledData0.txt
```

If the file has already been converted and you want to skip conversion:

```bash
mpirun -np 4 src/fastSF.out -U data/SampledData0.h5
```

If the sampled-data file is in `in/`, the corresponding commands are:

```bash
mpirun -np 4 src/fastSF.out -U SampledData0.txt
```

or

```bash
mpirun -np 4 src/fastSF.out -U SampledData0.h5
```

The input resolver checks the provided path first, then common local locations such as `in/` and `data/`.

## `Processors_X` Guidance

Choose `Processors_X` so that:

1. `(Nx / 2)` is divisible by `Processors_X`
2. `(Ny / 2)` is divisible by `P / Processors_X`

For a periodic `65 x 65 x 65` sampled-data file, `fastSF` trims the duplicated endpoint and computes on a `64 x 64 x 64` grid. That means:

- `Nx / 2 = 32`
- `Ny / 2 = 32`

Good choices are:

- `P = 2`  -> `Processors_X = 2`
- `P = 4`  -> `Processors_X = 2`
- `P = 8`  -> `Processors_X = 4`
- `P = 16` -> `Processors_X = 4`
- `P = 32` -> `Processors_X = 8`
- `P = 64` -> `Processors_X = 8`

## Real Sample Result

The repository sample `turbulence_post_process/data/SampledData0.txt` was validated through this workflow using a copied input file. After periodic trimming, the generated structured HDF5 had:

- velocity field shape: `64 x 64 x 64`
- grid lengths: `64, 64, 64`
- total KE printed by `fastSF`: `32768`

## Practical Advice

- Use `.txt` only when you need conversion from sampled-data output.
- Use `.h5` for repeat runs so conversion is skipped.
- Keep `Only_longitudinal: true` for an initial production pass unless you need transverse structure functions.
- For large production runs, keep a backup of the original `.txt` if it is expensive to regenerate.
