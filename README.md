# CP_Variability_RAMSIN_IDEAL_Shear
This repository contains the RAMS model code and namelists used for the IDEAL-Shear simulations.

Unzip rams-6.3.02.zip

Enter the rams-6.3.02 directory

Run an initial RAMS simulation with the "BL_dev" version of the RAMSIN for the 100m or 250m resolution.

In src/6.2.02/micro there are three versions of mic_init.f90:

mic_init.f90, mic_init.f90.back_tracer_250m, and mic_init.f90.background_tracer_100m

To make the cold bubble the right shape (linear channel):
In the file "ruser.f90", comment out line 591 and delete the variable "acetmp4" from the equation in line 597.

Before running the namelist "RAMSIN.IDEAL-Shear-100m_Cold Pool" or "RAMSIN.IDEAL-Shear-250m_Cold Pool" create a copy of mic_init.f90.back_tracer_250m (for 250m simluation)
or mic.bachground_tracer_100m (for 100m simulation) called mic_init.f90 then recompile the modile with the modified mit_init.f90 and ruser.f90 files.

After the BL_dev RAMSIN simulation is finished, run the Cold_Pool Ramsin Simulation.

