# nekstat

Post-processing of **Nek5000 on-the-fly wing statistics** for a NACA-4412
turbulent boundary layer (Re<sub>c</sub> = 200 000, AoA = 5°). It reads the
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
├── src/nekstat/
│   ├── read_intfld.py   read_data_profiles.m  (binary read + all rotations + budgets)
│   ├── postproc.py      load_prof_data.m       (integral quantities, Cf/Cp, Cl/Cd)
│   ├── diagnostic.py    diagnostic.m + trapezoidal2.m + MATLAB-style interp1
│   ├── export.py        export_drag_decomposition.m + Re200k .mat writer
│   ├── naca_geom.py     naca_prof2 / create_profiles / rot / save_inter_mesh
│   ├── pipeline.py      high-level run() / process()
│   ├── result.py        Result accessor
│   ├── cli.py           `nekstat` command
│   └── viz/             velocity / budget / streamwise plots (single + overlay)
├── data/                sample case: ZSTAT/int_fld, x.fort, y.fort, int_mesh.mat,
│                        reference/stat_benchmark_N11_naca4412_200k.mat
├── examples/run_pipeline.py
└── docs/                figures for this README
```

## Mesh generation

`nekstat mesh` / `nekstat.build_mesh()` reproduce the NACA-4412 wall-normal
interpolation mesh (83 chordwise stations × 190 wall-normal points) and write the
little-endian Fortran-record `x.fort` / `y.fort` that Nek5000 reads. The committed
mesh uses leading-edge clustering to `x/c = 0.295` with uniform spacing
`dx = 0.01625` thereafter.

## Relation to the original MATLAB code

Direct ports of `main_post_proc_prof.m` and its dependencies. Behavioural details
are preserved deliberately, including the wall-frame rotations, the boundary-layer
edge criterion (`|u'|/U / √H₁₂ < 0.02`), the `bottom(1)=top(1)` leading-edge alias,
and the default export window (profiles 26–80). The Fortran 77 decode/interpolation
stage (Nek5000) is unchanged and outside this package.

## License

MIT © 2026 Yuning Wang
