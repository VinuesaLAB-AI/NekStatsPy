# Mesh Interpolation Integration Progress

This log records the implementation work for the configurable interpolation-mesh
integration. Entries are appended as milestones are completed.

## 2026-08-27 — Step 1: baseline and design confirmation

- Reviewed the legacy MATLAB mesh drivers and the current Python/Nek workflow.
- Identified `data/int_mesh.mat` as the NACA4412 compatibility baseline:
  31,540 points, 83 chordwise stations, and 190 wall-normal points.
- Identified `data/meshes/naca0012/int_mesh.mat` as the NACA0012 baseline:
  137,120 points, 80 chordwise stations, and 857 wall-normal points.
- Confirmed that the current Python generator allocates an impractically large
  temporary grid for its nominal tolerance. The replacement will use direct,
  analytic NACA geometry and retain explicit legacy-compatible presets.
- Decided to keep copied meshes backward compatible while adding generated
  `x.fort`, `y.fort`, `int_mesh.mat`, a figure, and a manifest as one artifact
  set.

## 2026-08-27 — Step 2: memory-safe configurable generator

- Replaced the sampled-beta implementation with analytic NACA-4 surface and
  tangent evaluation, including a dedicated symmetric-airfoil (`NACA00xx`)
  branch.
- Added named compatibility presets for the committed NACA4412 reference, the
  currently uncommented NACA4412 MATLAB settings, and the supplied NACA0012
  reference.
- Added `InterpolationMesh` capability metadata, profile-layout validation,
  Cartesian-grid generation, MATLAB metadata export, mesh figures, and a YAML
  manifest with artifact checksums.
- Verified the generated NACA4412 and NACA0012 arrays against their bundled
  `.mat` assets. Maximum coordinate deviations are respectively `6.3e-10` and
  `2.7e-10`; this is the analytic limiting difference from the MATLAB sampled
  spline, not a change in point layout.

## 2026-08-27 — Step 3: configuration, CLI, and run handoff

- Added the config-first `nekstat mesh [config.yml]` interface. It accepts a
  complete run configuration or a mesh-only YAML file and writes the complete
  artifact set to `mesh.output.directory` (or `--out`).
- Preserved the old no-config command as a NACA4412-reference convenience path.
- Extended `mesh.mode: generate` in the Nek run preparer to write the same
  self-consistent artifacts into the run directory.
- Added `post.mesh: auto`, which resolves only to that generated in-run
  `int_mesh.mat`; incompatible coordinate-only meshes are rejected before a
  profile post-processing run starts.
- Added a standalone NACA4412 reference generation configuration example.

## 2026-08-27 — Step 4: small Nek benchmark validation

1. Added `configs/naca4412_mesh_benchmark.yml`, using benchmark files 2–6.
   File 1 is used only to obtain the exact start time, so no unsupported
   estimate of the first accumulation interval is made.
2. Ran `nekstat run configs/naca4412_mesh_benchmark.yml --prepare-only` and
   verified `p068=6`, `p069=2`, and `p070=4.500170000014` in the staged `.rea`.
3. Ran the Nek interpolation with generated `naca4412_reference` artifacts
   and no full Python post-processing. Nek completed in 3.47 seconds and wrote
   a 31,540-point `int_fld` (66 statistics and 68 derivatives).
4. Calculated one block value directly from the generated `int_fld` at AoA 5°:
   `Cl=0.8989107320225`, `Cd=0.02021556397457`,
   `Cd_p=0.006751644929217`, and `Cd_f=0.01346391904535`.
5. Recomputed from the same field using the bundled reference metadata. The
   differences were `|dCl|=7.67e-12`, `|dCd|=8.44e-11`,
   `|dCd_p|=9.45e-14`, and `|dCd_f|=8.43e-11`.

Nek reported one interpolation point within `2.1e-11` of a boundary, but wrote
the configured completion marker and a valid output file; this is consistent
with the reference-compatible leading/trailing-edge layout.

## 2026-08-27 — Step 5: cross-variant verification and documentation

- Ran the config-first CLI for `naca0012_reference`; it produced 137,120 points
  (80 × 857) and matched the bundled asset within `2.7e-10` in coordinates.
- Ran the Cartesian far-field example; it produced 100,651 coordinate-only
  points and correctly omitted profile metadata.
- Added regression tests for reference arrays, Fortran/MATLAB artifact
  consistency, output-path safety, Cartesian point order, and generated-mesh
  run staging with `post.mesh: auto`.
- Ran the complete suite: 25 tests passed.
- Updated the README and the NACA0012 status note to document the generated
  workflow and remove the obsolete symmetric-profile limitation.
