# nekstat

Post-processing of **Nek5000 on-the-fly wing statistics** for NACA airfoil
turbulent boundary-layer cases. The bundled production templates cover
NACA-4412 at Re<sub>c</sub> = 200 000, AoA = 5°, and NACA-0012 at
Re<sub>c</sub> = 200 000, AoA = 0°. A parallel KTH Framework/PSTAT2D workflow
covers the `no_resize_lc` channel-flow case at Reb = 4600. The airfoil pipeline reads the
interpolated statistics field `ZSTAT/int_fld`, rotates every quantity into the
wall-tangent / wall-normal frame, and produces boundary-layer profiles, integral
quantities, the full Reynolds-stress / turbulent-kinetic-energy budget, and the
C<sub>l</sub>/C<sub>d</sub> decomposition.

This is a pure NumPy/SciPy port of the original MATLAB pipeline and reproduces its
output to **float64 round-off** (see [Validation](#validation)).

---

## What it does

The simulation dumps span- and time-averaged statistics (`sts*`); a Nek5000
post-processing run spectrally interpolates them onto a wall-fitted mesh, giving
`ZSTAT/int_fld` (66 statistics + 68 derivatives per point). `nekstat` takes it
from there:

```
int_fld + int_mesh.mat                     (this package)
        │  read_data_profiles   rotate mean flow, Reynolds stresses, gradients,
        │                       and the full TKE budget into the wall frame
        │  load_prof_data       inner scaling, BL edge (diagnostic), Re_θ / Re_τ /
        │                       H₁₂ / β / Cf / Cp, budget scalars, Cl/Cd
        │  export               nested Re200k struct  ->  stat_*.mat
        ▼
   velocity profiles · TKE budgets · streamwise-developing quantities
```

## Install

```bash
pip install git+https://github.com/yuning-w/nekstat.git
# or, from a clone (editable, for development):
git clone https://github.com/yuning-w/nekstat.git
cd nekstat && pip install -e .
```

Requires Python ≥ 3.8, NumPy, SciPy, Matplotlib (pulled in automatically). The
sample dataset lives in the repository under `data/` (it is intentionally not
bundled into the wheel because `int_fld` is ~33 MB) — use a clone to run the
examples, or point the tools at your own Nek output.

## Quick start

### Command line

```bash
# close the loop from ordered Nek sts files -> int_fld -> stat .mat
# edit source.path in the config first
nekstat run configs/naca4412_post.yml
nekstat run configs/naca0012_post.yml

# run fixed-end backward windows and write a Python-only Cl/Cd convergence CSV
nekstat clcd-backward configs/naca4412_post.yml
nekstat clcd-backward configs/naca0012_post.yml

# stage a run folder without launching Nek or Python post-processing
nekstat run configs/naca4412_post.yml --prepare-only

# stage the backward Cl/Cd run folder and write the planned window manifest
nekstat clcd-backward configs/naca4412_post.yml --prepare-only

# rebuild the case executable with the local MPI/compiler environment
nekstat build configs/naca4412_post.yml --rebuild
nekstat build configs/naca0012_post.yml --rebuild
nekstat build configs/channel_lc_post.yml --rebuild

# rebuild first, then stage/run/post from the same config
nekstat run configs/naca4412_post.yml --build --rebuild

# KTH PSTAT2D channel workflow
nekstat run configs/channel_lc_post.yml
nekstat channel-post configs/channel_lc_post.yml --int-fld tmp/no_resize_lc/int_fld --out tmp/result_data/channel_python.mat --no-figures
nekstat validate-channel --mine channel_python.mat --ref matlab_reference.mat

# process int_fld -> stat .mat   (defaults to the bundled sample case)
nekstat post --nek data --mesh data/int_mesh.mat --out stat.mat

# (re)generate the interpolation mesh and check it against int_mesh.mat
nekstat mesh --ref data/int_mesh.mat

# plot the three figure categories for one dataset
nekstat plot --mat stat.mat --figdir figs

# overlay + machine-precision diff against a reference result
nekstat validate --mine stat.mat --ref data/reference/stat_benchmark_N11_naca4412_200k.mat
```

### Python API

```python
import nekstat
from nekstat import viz

# full pipeline -> Result (and optionally a .mat)
res = nekstat.run("data", "data/int_mesh.mat", out_file="stat.mat")

res.scal("ss", "Cf")          # suction-side skin friction vs streamwise station
res.prof("ss", res.station("ss", 0.75), "U")   # U profile at x/c = 0.75
res.scal("ss", "Cl")[0]       # 0.87120

viz.plot_single(res, "figs/")                       # solid-line figures
ref = nekstat.load_result("data/reference/stat_benchmark_N11_naca4412_200k.mat")
viz.plot_overlay(ref, res, "validation_figs/")      # overlay two datasets
```

## Output structure

`stat_*.mat` holds a `Re200k` struct with `top.ref` (suction side) and `bot.ref`
(pressure side), each a `1 × Nprof` struct array of profiles. Per profile:

| group | fields |
|---|---|
| geometry | `xa, ya, alpha, x, y, yn` |
| mean flow | `U, V, W, P, pp, dPdx, dPdy` |
| Reynolds stresses | `uu, vv, ww, uv, uw, vw` (+ inner-scaled `uup…uvp`) |
| scaling / edge | `ut, tauw, Ue, Uinf, delta99, deltas, theta, i99` |
| integral | `Cf, Cp, beta, Reth, Ret, H12` |
| TKE budget (inner) | `Pkp, Dkp, Tkp, VDkp, VPGkp, Ckp` |
| aerodynamics | `Cl, Cd, Cd_p, Cd_f, LD` |

The `Result` class wraps this: `.scal(side, field)`, `.prof(side, p, field)`,
`.station(side, x_c)`, `.yplus(side, p)`, with `side ∈ {"ss","ps"}`.

## Validation

Against the original MATLAB output (`stat_benchmark_N11_naca4412_200k.mat`,
same `int_fld`), every quantity agrees to float64 round-off:

| quantity | max &#124;Python − MATLAB&#124; | quantity | max diff |
|---|---|---|---|
| U, V | 2×10⁻¹⁶ | Cf | **0** (exact) |
| uu | 3×10⁻¹⁸ | β / Reθ | 4×10⁻¹⁴ / 1×10⁻¹² |
| TKE budget terms | ~1×10⁻¹⁶ | Reτ / H₁₂ | 3×10⁻¹⁴ / 1×10⁻¹⁵ |

The interpolation mesh generator reproduces `int_mesh.mat` to ~1×10⁻⁹ and its
Fortran `x.fort`/`y.fort` writer is byte-exact.

Overlay figures (MATLAB reference = lines, Python port = markers):

![Velocity profiles](docs/velocity_profiles.png)
![TKE budget](docs/tke_budget.png)
![Streamwise quantities](docs/streamwise.png)

## Package layout

```
nekstat/
├── configs/             config-driven run templates
├── src/nekstat/
│   ├── read_intfld.py   read_data_profiles.m  (binary read + all rotations + budgets)
│   ├── postproc.py      load_prof_data.m       (integral quantities, Cf/Cp, Cl/Cd)
│   ├── diagnostic.py    diagnostic.m + trapezoidal2.m + MATLAB-style interp1
│   ├── export.py        export_drag_decomposition.m + Re200k .mat writer
│   ├── naca_geom.py     naca_prof2 / create_profiles / rot / save_inter_mesh
│   ├── pipeline.py      high-level run() / process()
│   ├── nek5000.py       config -> runs/<name> staging + Nek launch + post hook
│   ├── clcd_backward.py fixed-end backward Cl/Cd convergence sweeps
│   ├── kth_channel.py   KTH Framework/PSTAT2D channel staging + run hook
│   ├── channel_intfld.py/channel_pipeline.py
│   │                    KTH int_fld reader + MATLAB-free channel post
│   ├── result.py        Result accessor
│   ├── cli.py           `nekstat` command
│   └── viz/             velocity / budget / streamwise plots (single + overlay)
├── data/                sample case: ZSTAT/int_fld, x.fort, y.fort, int_mesh.mat,
│                        reference/stat_benchmark_N11_naca4412_200k.mat,
│                        dependencies/<case> reusable Nek case assets,
│                        meshes/naca0012 copy-mode interpolation mesh
├── runs/                generated run workspaces (ignored by git)
├── examples/run_pipeline.py
└── docs/                figures for this README
```

## Config-Driven Nek Stage

`nekstat run <config.yml>` creates a reproducible run folder, saves the resolved
configuration as `current_conf.yml`, stages the selected case dependencies,
orders the selected source `sts*` window into `ZSTAT/stsnaca_wing0.fNNNNN`,
patches the Nek `.rea` statistics parameters, launches the case-local executable,
and then calls the existing Python post-processing pipeline.

The original Nek solver archive is preserved in `tmp/Nek1093_dong.tar.gz` for
portable rebuilds; the config-driven run path uses the precompiled executable in
`data/dependencies/naca4412/nek5000_4412`.

For a new machine or MPI stack, edit the config's `build.env_script` if needed
and run:

```bash
nekstat build configs/naca4412_post.yml --rebuild
```

This extracts `tmp/Nek1093_dong.tar.gz` into `tmp/Nek1093_dong` when needed,
stages the compile work directory from `data/dependencies/naca4412`, sources the
configured environment script, runs the case compile command in the configured
`build.work_dir`, and installs the resulting executable at the configured
`build.output` path.

When `nek.sampling_start_time` is omitted or set to `auto`, `nekstat` reads the
132-byte header of the preceding `sts` file and uses that end time for Nek
parameter `p070`. If `source.first_file` is `1`, there is no predecessor, so set
`nek.initial_start_time` explicitly, or set
`nek.allow_first_file_end_time_as_start: true` if that legacy zero-interval first
file behavior is intended. The run launcher also creates the Nek output
directory `A0`, clears stale `A0` field files, removes old `ZSTAT/int_fld` and
`*.sch*` files, and checks the Nek log for the configured completion marker.

The ready templates are:

| config | case | Nek source | mesh |
|---|---|---|---|
| `configs/naca4412_post.yml` | NACA4412, Re=200k, AoA=5 deg | `tmp/Nek1093_dong.tar.gz` | `data/ZSTAT`, `data/int_mesh.mat` |
| `configs/naca0012_post.yml` | NACA0012, Re=200k, AoA=0 deg | `tmp/nek1072.tar.gz` | `data/meshes/naca0012/` |
| `configs/channel_lc_post.yml` | channel `no_resize_lc`, Reb=4600 | `tmp/KTH_Framework.tar.gz` | `DATA/int_pos`, `DATA/c2Dtcf0.f00001` |

The main config fields are:

| section | purpose |
|---|---|
| `run` | run name and root folder, e.g. `runs/naca_wing_XXX` |
| `case` | case id, Nek session name, dependency folder, executable |
| `build` | solver archive, local MPI environment script, compile command, output executable |
| `mesh` | where to get `ZSTAT/x.fort` and `ZSTAT/y.fort`, or generate them |
| `source` | source result tree, glob, and 1-based window over sorted `sts*` files |
| `nek` | MPI launch settings and `.rea` parameter overrides |
| `post` | existing Python `int_fld -> stat_*.mat` options |
| `clcd_backward` | Python-only fixed-end Cl/Cd convergence sweep options |
| `pstat2d` / `channel_post` | KTH PSTAT2D `tcf.par` patching and channel `.mat` export |

Use `--prepare-only` to stage and inspect the folder on a login node. Use
`--skip-nek` or `--skip-post` when one stage has already been completed.

For the NACA0012 template, the archived run used `source.first_file: 1` and
`nek.initial_start_time: 7.000000000028`. If you start from a later sorted
`sts` file, keep `nek.sampling_start_time: auto` and the launcher will read the
predecessor header. If your first selected file is still file 1 but begins at a
different physical time, update `nek.initial_start_time` in the config.

### Backward Cl/Cd convergence

`nekstat clcd-backward <config.yml>` uses the shared `case`, `build`, `mesh`,
`source`, and `nek` sections, then reads `clcd_backward` for the sweep-specific
window and output settings. The ordinary `nekstat run` command continues to use
the `post` section for the full `int_fld -> stat_*.mat` export.

The backward workflow keeps the final selected `sts` file fixed and advances the
first included file by `clcd_backward.stride`. For a sweep with
`first_file: 100`, `last_file: 199`, and `stride: 5`, the Nek parameters are
patched as windows `100-199`, `105-199`, `110-199`, and so on. For every window,
`p070` is inferred from the predecessor `sts` header, matching the single-run
`sampling_start_time: auto` behavior.

The run folder contains:

```text
runs/naca_wing_clcd_backward/
├── clcd_backward.csv       # Cl, Cd, Cd_p, Cd_f, L/D, and timing metadata
├── nek_windows.csv         # planned/generated/reused Nek windows
├── int_fld/                # per-window int_fld snapshots
├── logs/                   # per-window Nek logs
├── ZSTAT/                  # staged sts files and transient int_fld
└── A0/                     # transient Nek output directory
```

Existing valid per-window `int_fld` files are reused unless `--force` or
`clcd_backward.force: true` is set. `--skip-nek` recomputes the CSV from existing
`int_fld` snapshots. `--discard-int-fld` removes those snapshots after the CSV is
written. The coefficient extraction is pure Python and reads only the records
needed for Cl/Cd: `F4` and `D1-D4`.

### KTH channel workflow

`workflow: channel2d` selects the KTH Framework/PSTAT2D path. It stages the
`tcf` case assets, `DATA/int_pos`, `DATA/c2Dtcf0.f00001`, and selected source
`ststcf0.f*` files. Because this KTH case ignores `STS_FFILE`, selected source
files are renumbered into `DATA/ststcf0.f00001...`, and `_PSTAT2D:STS_NFILE`
plus `_PSTAT2D:STS_STIME` are patched in `tcf.par`.

The MATLAB `post_STAT.m` step is replaced by `nekstat channel-post`, which reads
the KTH `int_fld` format directly, computes the same channel profile fields and
wall units, writes a MATLAB `data` struct, and can regenerate the same figure
set. `validate-channel` reports max absolute differences against a MATLAB
reference `.mat` when MATLAB is available on HPC.

Fresh channel parity check against MATLAB `post_STAT.m` output, focused on the
wall-unit velocity and Reynolds-stress profiles used for the channel case:

| field | max &#124;Python − MATLAB&#124; |
|---|---:|
| `Up` | 1.421e-14 |
| `uup` | 7.105e-15 |
| `uvp` | 5.551e-16 |
| `vvp` | 8.882e-16 |
| `wwp` | 1.554e-15 |

![Channel U+ MATLAB vs Python](docs/channel_uplus_comparison.png)

![Channel Reynolds-stress MATLAB vs Python](docs/channel_reynolds_stress_comparison.png)

## Mesh generation

`nekstat mesh` / `nekstat.build_mesh()` reproduce the NACA-4412 wall-normal
interpolation mesh (83 chordwise stations × 190 wall-normal points) and write the
little-endian Fortran-record `x.fort` / `y.fort` that Nek5000 reads. The committed
mesh uses leading-edge clustering to `x/c = 0.295` with uniform spacing
`dx = 0.01625` thereafter.

The NACA0012 workflow currently uses the supplied mesh files in
`data/meshes/naca0012/`: `ZSTAT/x.fort`, `ZSTAT/y.fort`, and `int_mesh.mat`.
That mesh has 80 profiles per side, 857 wall-normal points, and 137120 total
interpolation points. The symmetric-airfoil generated-mesh path is tracked in
`docs/naca0012_re200k_open_issue.md`.

## Relation to the original MATLAB code

Direct ports of `main_post_proc_prof.m` and its dependencies. Behavioural details
are preserved deliberately, including the wall-frame rotations, the boundary-layer
edge criterion (`|u'|/U / √H₁₂ < 0.02`), the `bottom(1)=top(1)` leading-edge alias,
and the default export window (profiles 26–80). The Fortran 77 decode/interpolation
stage (Nek5000) is unchanged; `nekstat run` now stages and launches it before
handing `ZSTAT/int_fld` to the Python pipeline.

## License

MIT © 2026 Yuning Wang
