"""
End-to-end example on the bundled sample case (NACA-4412, Re=200k, AoA=5 deg).

    python examples/run_pipeline.py

Processes ZSTAT/int_fld, writes a stat .mat, plots the three figure categories,
and validates against the MATLAB reference output.
"""

import os
import nekstat
from nekstat import viz

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)

paths = nekstat.sample_paths()

# 1. process int_fld -> Re200k
out_mat = os.path.join(OUT, "stat_benchmark_naca4412_200k.mat")
res = nekstat.run(paths["nek_dir"], paths["mesh"], out_mat)
print("Cl=%.5f Cd=%.5f L/D=%.2f"
      % (res.scal("ss", "Cl")[0], res.scal("ss", "Cd")[0],
         res.scal("ss", "Cl")[0] / res.scal("ss", "Cd")[0]))

# 2. plot the three categories for this dataset
viz.plot_single(res, os.path.join(OUT, "figs"))

# 3. overlay against the MATLAB reference
ref = nekstat.load_result(paths["reference"])
viz.plot_overlay(ref, res, os.path.join(OUT, "validation_figs"))
print("done -> outputs/")
