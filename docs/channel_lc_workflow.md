# KTH Channel Workflow

This note records the `no_resize_lc` channel-flow integration.

## Scope

The channel case uses KTH Framework/PSTAT2D rather than the NACA wing
interpolation case:

- Config: `configs/channel_lc_post.yml`
- Case assets: `data/dependencies/no_resize_lc/`
- Framework archive: `tmp/KTH_Framework.tar.gz`
- Local source drop: `tmp/no_resize_lc/` (ignored by git)

The KTH run reads `DATA/ststcf0.fNNNNN`, `DATA/c2Dtcf0.f00001`, and
`DATA/int_pos`, then writes `int_fld` in the run root.

## MATLAB Reference

In this workspace MATLAB is available through the direct executable
`~/apps/matlab/matlab`. The `matlab` alias in `~/.bashrc` is interactive-only,
so automation should call the executable directly.

The original `post_STAT.m` contains a stale hardcoded `pyenv` path. For the
reference generated here, a local ignored wrapper `tmp/no_resize_lc/post_STAT_local.m`
changed only that `pyenv` line to:

```matlab
pyenv(Version="/home/yuninw/apps/mpi_drl/miniforge3/envs/matlab/bin/python3.10")
```

Then the MATLAB reference was generated with:

```bash
cd tmp/no_resize_lc
mkdir -p ../result_data Figs
~/apps/matlab/matlab -batch "post_STAT_local"
```

The script writes:

```text
tmp/result_data/LC_Reb4600_bdfd_y0_amp0.mat
```

## Python Workflow

Run the full Nek/PSTAT2D/Python loop after setting `source.path`:

```bash
nekstat run configs/channel_lc_post.yml
```

Run only the MATLAB-free post step for an existing `int_fld`:

```bash
nekstat channel-post configs/channel_lc_post.yml \
  --int-fld tmp/no_resize_lc/int_fld \
  --out tmp/result_data/LC_Reb4600_bdfd_y0_amp0_python.mat \
  --no-figures
```

Compare against the MATLAB reference:

```bash
nekstat validate-channel \
  --mine tmp/result_data/LC_Reb4600_bdfd_y0_amp0_python.mat \
  --ref tmp/result_data/LC_Reb4600_bdfd_y0_amp0.mat
```

## Validation Notes

The channel postprocessor mirrors `post_STAT.m` for:

- Matrix fields: `x`, `yy`, `Ur`, `Vr`, `dUdyr`, `uur`, `vvr`, `wwr`, `uvr`,
  `Pr`, `pp`
- Averaged vectors: `y`, `U`, `V`, `dUdy`, `uu`, `vv`, `ww`, `uv`, `urms`,
  `prms`, `vvv`
- Wall-unit and integral fields: `twall`, `utau`, `Ret`, `lstar`, `yp`, `Up`,
  `Vp`, `uup`, `vvp`, `wwp`, `uvp`, `urmsp`, `Uinf`, `theta`, `deltas`,
  `Reth`, `H12`

A direct comparison between the MATLAB reference above and `nekstat
channel-post` output from the same `tmp/no_resize_lc/int_fld` gives roundoff
agreement for the wall-unit velocity and Reynolds-stress profiles. Selected
differences:

| field | max abs diff |
|---|---:|
| `Up` | `1.421e-14` |
| `uup` | `7.105e-15` |
| `uvp` | `5.551e-16` |
| `vvp` | `8.882e-16` |
| `wwp` | `1.554e-15` |

![Channel U+ MATLAB vs Python](channel_uplus_comparison.png)

![Channel Reynolds-stress MATLAB vs Python](channel_reynolds_stress_comparison.png)
