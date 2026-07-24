# Integrate Nek5000 usage into the workflow to enclose the pipeline 

# Background 
The codes here perfectly reproduces the partial pipeline of producing statistics files. In addition to it, I would like to incorporate the post-processing on NEK5000 ends. 

# Objective 
The idea is to put the solver (./tmp/Nek1093_dong.tar.gz) and compile dependices (./tmp/01_NEK) in the source and make a pre-complied executable `./nek5000_4412` (as I might add other cases like channel flow or naca0012 cases later). So it can produce `int_fld` within the work path and be used by python codes for analysis. 

# Instruction: 
1. Please had look at the `./tmp/01_NEK` which inprinciple contains everything we need to set up the F77 side. 
2. Please pay extra attention on `./tmp/01_NEK/naca_wing.usr` that gives you a solid understand how Nek programe reads binary `sts` files. 
3. Place the shared dependicies such as `naca_wing.re2` and `naca_wing.map` in `data/dependicies/naca4412` so it will be distinct 
4. There are some shell scripts `.sh` should give you instruction on how does the pipe line work.  

# Notes 
+ Please always preserve the `tar.gz` of Nek1093_dong as it conatins some built depencies which can be directly used when initialize this workpath instead of rebuild it. When the code is used elsewhere, it ensures the Solver is protable and appliable 

+ As I said, please ensure the flexibiliy for future incorporation for diverse cases. 

+ Unfortunately there is no data locally we can test the completed pipeline. But I will use it on HPC and get back to you. Overall I think it is a matter of path and file management rather than developing. The hardest part has been done here. 

# Other notes left from the latest conversation: 
- The whole `nekstat` build, the parameter gotcha (committed `int_mesh.mat` used `xn=0.295, dx=0.01625` → 83 stations), and the machine-precision validation are saved in my project memory, so I'll have the context even in a fresh session.

- For the F77 side, it'll help to point me at the Nek5000 post-processing branch / the `naca_wing.usr` + `ZSTAT` decode routines (in `01_NEK/`) so we can wire the `sts → int_fld` decode+interpolation stage into the package's workflow — e.g. a `nekstat` step that shells out to the compiled tool, or documenting the compile/run path alongside the Python pipeline.