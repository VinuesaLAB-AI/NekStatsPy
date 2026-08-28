# Interpolation-Mesh Optimization Test Log

This log records controlled tests seeking an interpolation mesh whose blockwise
NACA4412 force coefficients are close to the supplied long-time backward-average
reference. It distinguishes mesh-discretization effects from the much shorter
available benchmark averaging window.

## Target values and acceptance metric

The screenshot supplies the following approximate long-time target at AoA 5°:

| Quantity | Target |
|---|---:|
| `Cl` | 0.867 |
| `Cd_f` | 0.0128 |
| `Cd_p` | 0.0087 |
| `Cd` | 0.0215 |
| `L/D` | 41 |

For each quantity, the reported relative deviation is
`abs(candidate - target) / abs(target)`. The working target is no more than 5%
per reported quantity; because the reference values are rounded in the image,
this is an engineering comparison rather than a bitwise regression criterion.

## Test design

The short benchmark supplies six `sts` files ending at times 4.50017 through
5.17517, a 0.75-time-unit window. The screenshot is stated to use more than 40
time units, so agreement cannot be attributed solely to mesh selection.

To include file 1 safely, its accumulator start is inferred as
`4.425170000013`: file 1 has 5,000 samples and files 1–2 are separated by
0.075, while file 2 has 10,000 samples. The validation config records this
explicitly instead of using the invalid legacy convention of assigning file 1's
end time as its start time.

Planned controlled cases all use files 1–6 and the same Nek executable:

1. **Baseline** — `naca4412_reference`, 83 surface stations × 190 normal
   points. Establish the effect of including file 1.
2. **Chordwise refined** — retain wall-normal points; increase surface-station
   density and trailing-edge coverage. Cl/Cd integration uses surface values,
   so this is the mesh direction expected to affect the force quadrature.
3. **Leading-edge refined** — add leading-edge station density without changing
   the far-field/wall-normal extent.
4. **Best candidate confirmation** — rerun the selected configuration and
   compare generated-mesh and reference-metadata force extraction.

Wall-normal-only refinement is intentionally not a primary force test: the
lightweight Cl/Cd extractor uses pressure and velocity-gradient values at the
wall interpolation points, so adding off-wall points does not change the force
quadrature.

## Results

All cases completed Nek interpolation successfully using the same six files,
the explicit `4.425170000013` start time, and AoA 5°. “Unstable” indicates a
nonphysical wall-shear outlier caused by the new station placement; those cases
are not candidates for production use.

| Case | Stations × normal | `Cl` dev. | `Cd_f` dev. | `Cd_p` dev. | `Cd` dev. | `L/D` dev. | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline reference (`nle=40`, `dx=.01625`) | 83 × 190 | 3.69% | 5.48% | 22.55% | 5.86% | 8.34% | historical baseline |
| chordwise refined (`40`, `.008`) | 128 × 190 | 3.30% | 4.54% | 40.64% | 13.74% | 17.78% | reject |
| leading-edge refined (`80`, `.01625`) | 123 × 190 | 3.93% | 51.78% | 6.80% | 33.58% | 53.91% | reject |
| balanced fine (`80`, `.008`) | 168 × 190 | 3.79% | 5.97% | 16.31% | 3.05% | 5.29% | near target |
| balanced (`80`, `.010`) | 150 × 190 | 3.81% | 5.72% | 12.08% | 1.48% | 3.63% | near target |
| balanced (`80`, `.012`) | 138 × 190 | 3.79% | 5.29% | 8.91% | 0.46% | 2.55% | near target |
| balanced (`80`, `.014`) | 130 × 190 | 3.66% | 4.38% | 8.27% | 0.74% | 2.72% | near target |
| balanced (`80`, `.01425`) | 129 × 190 | 3.55% | 3.53% | 7.70% | 1.01% | 2.88% | near target |
| **recommended (`80`, `.0145`)** | **128 × 190** | **3.14%** | **1.07%** | **7.07%** | **2.23%** | **3.76%** | **best stable candidate** |
| balanced (`80`, `.01475`) | 127 × 190 | 1.45% | 1853% | 6.36% | 1101% | 91.69% | unstable; reject |
| balanced (`80`, `.015`) | 127 × 190 | 3.48% | 303% | 8.93% | 177% | 63.27% | unstable; reject |
| `.010` plus explicit trailing edge | 151 × 190 | 3.81% | 5.86% | 13.33% | 1.90% | 4.09% | no benefit |
| leading-count sweep (`82`–`90`, `.0145`) | 130–138 × 190 | 3.42–4.35% | 41.8–381.7% | 6.03–6.84% | 22.3–229.7% | 16.5–179.1% | unstable; reject |

## Recommendation and interpretation

Use [the recommended six-file config](../configs/naca4412_mesh_recommended.yml)
when the aim is this benchmark's force comparison. It specifies 80
leading-edge clustered stations, a `0.295` transition, `dx=0.0145`, and no
forced trailing-edge point. The confirmed block values are:

| `Cl` | `Cd_f` | `Cd_p` | `Cd` | `L/D` |
|---:|---:|---:|---:|---:|
| 0.8942612 | 0.0129368 | 0.0080849 | 0.0210216 | 42.5401 |

This meets the 5% target for `Cl`, `Cd_f`, total `Cd`, and `L/D`; `Cd_p` is
7.07% low. The controlled mesh search did not find a stable mesh that reaches
5% for every component. Because the available interval is 0.75 time units
versus the screenshot's more-than-40-unit backward average, that residual
pressure-drag difference should be treated primarily as sampling/convergence
uncertainty, not corrected by further mesh tuning.

The exact trailing edge and several nearby station patterns produced large,
unphysical skin-friction values. This makes a clear practical boundary: retain
the selected stable layout and improve the averaging horizon when additional
statistics become available, rather than forcing a mesh to match the rounded
long-time target.

## Follow-up: Cl and L/D-focused search

The requested follow-up narrows the objective to `Cl` and `L/D`, aiming for a
1–2% deviation while retaining finite, physically plausible drag components.
The starting layout is the recommended stable case (`nle=80`, `dx=.0145`).

The first controlled sweep varies only `transition_x`, which relocates the
join between leading-edge geometric clustering and uniform downstream spacing.
It keeps the same number of leading-edge stations, wall-normal grid, averaging
window, solver, and force-extraction method. Results will be appended below.

| Follow-up case | Stations × normal | `Cl` dev. | `L/D` dev. | Notes |
|---|---:|---:|---:|---|
| `transition_x=.26`, `nle=80`, `dx=.0145` | 131 × 190 | 3.79% | 25.08% | unstable drag phase; reject |
| `.28`, `80`, `.0145` | 129 × 190 | 3.73% | 34.00% | unstable drag phase; reject |
| `.31`, `80`, `.0145` | 127 × 190 | 3.73% | 2.25% | stable |
| `.33`, `80`, `.0145` | 126 × 190 | 3.80% | 2.87% | stable |
| **`.31`, `80`, `.0147`** | **126 × 190** | **3.68%** | **2.03%** | **best stable Cl/L-D trade-off** |
| `.31`, `80`, `.01475`–`.0149` | 126 × 190 | 3.67–3.68% | 2.18–2.76% | stable, but worse L/D |
| `.31`, `50`, `.0147` | 96 × 190 | 3.72% | 5.83% | under-resolved downstream quadrature |
| `.31`, `60`, `.0147` | 106 × 190 | 3.77% | 3.54% | stable |
| `.31`, `70`, `.0147` | 116 × 190 | 3.78% | 2.39% | stable |
| `.31`, `75`, `.0147` | 121 × 190 | 3.76% | 2.08% | stable |

The best Cl/L-D candidate gives `Cl=0.8989309`, `Cd_f=0.0133557`,
`Cd_p=0.0081329`, `Cd=0.0214886`, and `L/D=41.8329`. The corresponding
configuration is [naca4412_mesh_clld_candidate.yml](../configs/naca4412_mesh_clld_candidate.yml).
It was then rerun directly from that committed configuration and reproduced
`|dCl|=3.6829%` and `|d(L/D)|=2.0315%`.

The stable Cl range across the follow-up is `0.89884–0.89991`, while the target
is `0.867`. Because only unstable placements moved Cl toward the target, a
1–2% Cl deviation cannot be justified as a mesh improvement with these six
files. The defensible next lever is a longer statistics window, not a more
aggressive interpolation layout.
