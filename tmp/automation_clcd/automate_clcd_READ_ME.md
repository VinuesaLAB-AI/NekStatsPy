# Backward-averaged Cl/Cd automation

`automate_clcd.py` keeps the final `sts` file fixed, advances the first
included file by a configurable stride, runs the existing Nek postprocessor
in an isolated workspace, and extracts Cl/Cd with one MATLAB batch.

The source `.rea`, source `ZSTAT`, and existing `ZSTAT/int_fld` are never
modified. Generated `.rea`, `SESSION.NAME`, `logfile`, `A0` fields, and
`int_fld` files live below the selected output directory.

## Environment

Load MATLAB before starting a full or MATLAB-only run. On the current JUWELS
setup:

```bash
source ~/run-Matlab.sh
```

The Nek wrapper loads its own OpenMPI/UCX environment in its subprocess.

## Inspect the planned windows

Always begin with a dry run:

```bash
./automate_clcd.py \
    --first-file 100 \
    --last-file 199 \
    --stride 5 \
    --aoa-deg 5 \
    --dry-run
```

`--last-file` and `--aoa-deg` are required because both are case-specific.
For this example the windows are `100-199`, `105-199`, `110-199`, and so on.

For every window, `p070` is taken from the end time in the header of the
preceding `sts` file. If the first requested file has no preceding file in
`ZSTAT`, provide the sampling start explicitly:

```bash
./automate_clcd.py \
    --first-file 1 \
    --last-file 199 \
    --stride 5 \
    --aoa-deg 5 \
    --initial-start-time 25.5 \
    --dry-run
```

The driver rejects missing indices, non-increasing header times, ambiguous
start times, and invalid ranges before launching Nek.

## Run the complete workflow

```bash
source ~/run-Matlab.sh

./automate_clcd.py \
    --first-file 100 \
    --last-file 199 \
    --stride 5 \
    --aoa-deg 5
```

The default output directory encodes the case, final file, stride, and AOA.
Use `--output-dir PATH` to choose it explicitly.

Important outputs are:

```text
<output-dir>/
├── <case-name>_clcd_backward.csv
│                          # One result row per backward window
├── nek_windows.csv         # Window, timing, file, and status manifest
├── run_config.json         # Reproducible command configuration
├── int_fld/                # Renamed per-window int_fld files
├── logs/                   # Nek wrapper, Nek logfile, and MATLAB logs
└── work/                   # Isolated reusable Nek workspace
```

The coefficient CSV includes the file range, actual averaging times,
Reynolds number, AOA, pressure reference, `Cl`, `Cd`, `Cd_p`, `Cd_f`, and
`L_over_D`.

## Resume and split stages

Valid existing `int_fld` outputs are reused automatically. Their header
start/end times are checked against the requested window. Use `--force` to
regenerate them.

Run only Nek:

```bash
./automate_clcd.py \
    --first-file 100 --last-file 199 --stride 5 --aoa-deg 5 \
    --stage nek --output-dir PATH
```

Later, load MATLAB and extract the coefficients:

```bash
source ~/run-Matlab.sh

./automate_clcd.py \
    --first-file 100 --last-file 199 --stride 5 --aoa-deg 5 \
    --stage matlab --output-dir PATH
```

`--discard-int-fld` removes the accumulated binary files only after a
successful MATLAB export. Keeping them is the default and is recommended for
restartability and auditability.

Run `./automate_clcd.py --help` for path overrides and all options.

## Minimal MATLAB calculation

`compute_clcd_from_int_fld.m` reads only:

- `F4`: mean pressure
- `D1-D4`: in-plane mean-velocity gradients

It seeks past all other statistical and derivative records, reproduces the
existing top/bottom rotations and `trapezoidal2.m` force integration, and
uses the AOA passed by the driver. The batch entry point is
`compute_clcd_batch.m`.
