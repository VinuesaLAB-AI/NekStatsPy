# NACA0012 Re200k Usage and Remaining Issue

This note records the current status of the NACA0012 Re=200k, AoA=0 deg case.

## Status

The case is now usable through the config-driven copy-mesh workflow:

- Run template: `configs/naca0012_post.yml`
- Nek case assets: `data/dependencies/naca0012/`
- Interpolation mesh assets: `data/meshes/naca0012/`
- Local mesh-generation source drop: `tmp/00_GenProfile/` (ignored by git)
- Local produced mesh drop: `tmp/0012_gen_meshes/` (ignored by git)

The supplied mesh assets contain:

- `data/meshes/naca0012/ZSTAT/x.fort`
- `data/meshes/naca0012/ZSTAT/y.fort`
- `data/meshes/naca0012/int_mesh.mat`

The `int_mesh.mat` metadata reports 80 profiles per side, 857 wall-normal
points, and 137120 total interpolation points. This matches the archived
NACA0012 Nek run log from `tmp/naca0012_200k_compile/LogRun.txt`, which reports
`found 137120 points in interp. mesh`.

## Nek Compatibility

The Nek-side post-processing case in `tmp/naca0012_200k_compile/` remains
compatible with the existing `nekstat` workflow:

- The session name is still `naca_wing`.
- Source statistics are expected as `ZSTAT/stsnaca_wing0.fNNNNN`.
- `.rea` parameters keep the same meaning:
  - `p068`: last/stat target file
  - `p069`: first included stat file
  - `p070`: sampling start time
- The `.usr` writes `ZSTAT/int_fld` with `nstats=66`, `nderiv=68`, and the
  completion marker `Output interporlated field in int_fld`.
- The run uses the same `A0` output directory behavior now handled by
  `nekstat.run`.

The archived run used `p068=107`, `p069=1`, and
`p070=7.000000000028`. The template keeps `last_file: auto`, so the launcher
patches `p068` from the matched source files. When `first_file` is greater than
1, `sampling_start_time: auto` reads the predecessor `sts` header. When
`first_file` is 1, update `nek.initial_start_time` if the production data starts
at a different physical time.

## Remaining Backend Issue

`nekstat.naca_geom.build_mesh("NACA0012")` is still not production-ready.
The current Python generator is tuned for the NACA4412 mesh and `naca_prof2()`
does not yet handle the symmetric-profile `p=0` case. Also, `mesh.mode:
generate` does not expose the NACA0012 sampling controls from
`tmp/00_GenProfile/A1_create_int_mesh_wing.m`, including:

- `yn = 0.12`
- `r = 1.002`
- `nle = 40`
- `dx = 0.6 / nle`
- leading-edge transition location `xn = 0.395` in
  `create_profiles_wing_testbudget.m`

For now, use `mesh.mode: copy` with the supplied mesh assets. A later backend
upgrade can generalize symmetric NACA profile generation and expose those mesh
sampling parameters in config.
