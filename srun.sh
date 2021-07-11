# !/bin/bash
echo "[srun] $@"
srun -G 4 --nodelist=104server -p rtx2080 --mem 30G --pty $@
