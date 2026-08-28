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

## Generated-Mesh Support

The generated-mesh backend now supports the symmetric NACA0012 case through
the `naca0012_reference` preset. It explicitly handles the `p=0` symmetric
profile rather than following the `0/0` camber calculation in the legacy
MATLAB `naca_prof2.m` implementation.

The preset exposes the archived sampling fingerprint:

- `target_extent = 0.12`
- `growth_ratio = 1.002`
- `leading_edge_count = 40`
- `uniform_dx = 0.015` (`0.6 / 40`)
- `transition_x = 0.395`

Use `nekstat mesh configs/mesh_naca0012_reference.yml` to produce a matching
artifact directory, or set `mesh.mode: generate`, `mesh.preset:
naca0012_reference`, and `post.mesh: auto` in a full run configuration. The
generated coordinates agree with the supplied MATLAB asset within `1e-8` and
the generated Fortran coordinates are tested against their generated `.mat`
metadata. The existing copy-mode run template remains supported for archived
workflows.
