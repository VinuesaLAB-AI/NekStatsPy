# Automation of Backward-Averaging of the statistical data for checking the convergence of Cl and Cd 

# Main path: 
`/p/scratch/deepwing/yuningw/03-DRL-TCF-WING/03-PostProcess/02-WING-Post/010-Statistics-WING`

# Background 
## Framework:
Here is a pipeline of post-processing of on-the-fly statistics data for turblent flow simulation of a wing section in Nek5000:
There are three stages:
+ Stage A — mesh generation (00_GenProfile/wrt_int_mesh_wing.m): builds the NACA-4412 wall-normal interpolation grid and writes x.fort/y.fort (Fortran binary) plus int_mesh.mat. I verified the on-disk x.fort/y.fort match int_mesh.mat to 0.0 once the header is read correctly (4 leading int32 words: eor, npoints, eor, eor). Grid = 83 chordwise stations × 190 wall-normal points, top + bottom (npoints=31540).

+ Stage B — processing (main_post_proc_prof.m → read_data_profiles → load_prof_data → export_drag_decomposition):
Read int_fld (Re=2e5, nstat=66, nderiv=68, 31540 pts). Fields are laid out column-major per profile with an 8-byte gap between records (first record has no gap).
Rotate every vector/tensor into wall-tangent/normal coordinates using per-station alpha (top uses [[c,s],[-s,c]], bottom uses [[c,s],[s,-c]]; bottom(1) aliases top(1) at the leading edge, last column unused).
Compute Reynolds stresses uu,vv,ww,uv, pressure stats, velocity/pressure gradients.
BL edge via diagnostic.m (the |u'|/U / √H12 < 0.02 criterion) → delta99, theta, deltas, Ue, i99, then Reth, Ret, H12, beta, Cf, Cp.
Cl/Cd/Cd_p/Cd_f/LD via trapezoidal2.m surface integration over all 83 stations.

+ Stage C — visualization: 01-Integral-Quantities.py, 02-WallNormal-Profiles.py, 03-clcd.py load outputs_data/stat_<case>_naca4412_200k.mat (a Re200k struct with top.ref/bot.ref struct-arrays) and read fields xa, Cf, beta, Reth, Ret, H12 (streamwise); yn, U, V, uu, vv, ww, uv, ut, nu, Ue (profiles); Cl, Cd, Cd_p, Cd_f, LD (aero).

## What I am doing: 
I wanted to check the lift and drag coefficient convergence by checking the backward average of the statistics. Meaning that I will always include the latest `sts` file when calculatiing the statistics but exclude the `sts` from the starting point gradually (see `01-NEK-Pstat/naca_wing.rea` LINE71~73), each time I calculate the Cl Cd, eventually to see where does the Cl Cd value converged. 
 
# Objective:
For that purpose, it would be very tedious to manually modify the configuration and then exectue matlab file back and forth. And also we only care about the Cl Cd value at the end of day instead of the completed statistics. 
So my idea is to automatize the working pipeline, the idea is not to change anything on the F77 Nek5000 side, instead, we deal with the matlab cleverly. 

Here is my design in details:
At `01-NEK-Pstat/`
1. Inspect `sts` files in `ZSTAT` to determine the starting time (`head`), how many files should I exclude each time and also the totall amount. 
2. A bash or python script that can automatically re-write the `.rea` config and execute the `01-NEK-Pstat/02-run-postNek` based on our configuration in a for loop. For each iteration, when it outputs an `int_fldXXX`, `mv` it and give it a new name. So that the `int_fld` files accumulate
At `02-Matlab-Read-Stat`
3. A for loop that reads accumulated `int_fldXXX` (take `02-MatLab-Read-Stat/matlab_func/read_data_profiles.m` as refence on how single `int_fld` is read) 
4. Instead of reading the full field, only reads the related quantities used for Cl Cd calculation (see very end of `02-MatLab-Read-Stat/matlab_func/load_prof_data.m`) and export the Cl Cd as a `.dat` or `.txt` file which is in ASCII. 

