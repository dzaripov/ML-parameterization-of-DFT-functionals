import os

import numpy as np
from mpmath import chebyt, chop, taylor

sbatch_template = """#! /bin/bash
#SBATCH --job-name="NN_{functional}"
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=schneider.mark14@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="/home/mmedvedev/schnm/log/NN_PBE_final/Final_No_dropout_Shifted_kappa_no_activation_Start_with_normal_LR_{functional}_0.2_{omega:.5f}_RMSE_"%j.out
#SBATCH --constraint="type_b|type_c"
#SBATCH --time=3-0
# Executable
CUBLAS_WORKSPACE_CONFIG=:16:8 python predopt_train.py --Name {functional} --N_preopt 10 --N_train 150 --Batch_size {batch_size} --Dropout 0.2 --Omega {omega:.5f} --LR_predopt 0.01 --LR_train 0.0001"""

functionals = [
    ("PBE_6_32", 1),
    ("XALPHA_6_32", 1),
]

n = 9
omegas = list(np.roots(chop(taylor(lambda x: chebyt(n, x), 0, n))[::-1]) / 2 + 0.5) + [
    0,
    1,
]

omegas = np.array(omegas)

for functional, batch_size in functionals:
    for omega in omegas:
        script = sbatch_template.format(
            functional=functional,
            omega=omega,
            batch_size=batch_size,
        )
        with open("run_calculations.slurm", "w") as file:
            file.write(script)
        os.system("sbatch run_calculations.slurm")
