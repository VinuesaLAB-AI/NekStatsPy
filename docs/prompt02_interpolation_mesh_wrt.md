# Generating interpolation mesh for wing statistics 

## Objective
The objective here is to provide various type of the interpolation mesh for different study. 
For the previous case, I put a generated interpolation mesh for NACA4412 and NACA0012 cases. But Now I need functions that can generate the new interpolation mesh for Nek-backend stats and also the nekstat calculation. 

## Background
In the `/home/yuninw/codes/post_process/nekstat/tmp/00_GenProfile` I am putting the original matlab function and scripts that used to generate the interpolation mesh in `.fort` and `.mat` format. This includes several cases that user can choose from and it dumps the file to the Nek / Matlab work space plus a figure showcasing the mesh grids.

## Instruction
1. Have a close look into the details of the matlab implementation. Get back to me the report including: 
    + How does the pipeline work?
    + What are the difference between cases usage? 
    + Where are the file dumped? 
    + What are the difference between differen functions for writting wing mesh? 
2. Using matlab, Replicate a mesh that is identical to `/home/yuninw/codes/post_process/nekstat/data/int_mesh.mat` and `/home/yuninw/codes/post_process/nekstat/data/ZSTAT` I belive I used the configuration uncommented in the script 
3. Migrate the functions into the python framework, then again replicate the same mesh. 
4. The expected functionality will be:
    + `.yml` configuration to control the mesh parameter, outputs direction, etc
    + manifest command `nekstat mesh` for generation
5. Make a validation test on a small chunck of data in `/home/yuninw/codes/post_process/nekpost/stats/naca4412/sts_data/benchmark`. Please focus on report and calculating the CL and CD here. No need to be backward averaging, just the block-wise value is fine.

## Toolkits
The mpi: `source ~/.bashrc.openmpi_ucx`
The python env use `source ~/.bashrc.miniforge`
The matlab use `~/apps/matlab/matlab  -nodesktop` also there is an alias in `~/.bashrc`
