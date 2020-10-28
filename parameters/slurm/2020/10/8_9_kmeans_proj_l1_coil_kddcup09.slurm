#!/bin/bash
#SBATCH --job-name=kmeans_proj_l1_coil_kdd          # nom du job
#SBATCH --partition=cpu                                         # partition GPU choisie
#SBATCH --ntasks=1                                                 # nombre de tache MPI (= nombre de GPU ici)
#SBATCH --ntasks-per-node=1                                        # nombre de tache MPI par noeud
#SBATCH --cpus-per-task=1                                          # nombre de coeurs CPU par tache (un quart du noeud ici)
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --time=20:00:00                                           # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=%j.out                                           # nom du fichier de sortie
#SBATCH --error=%j.err                                            # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=1-80                                             # array batch indices availabale wiht $SLURM_ARRAY_TASK_ID
#SBATCH --hint=nomultithread                                      # hyperthreading desactive
#SBATCH --mem=30GB

ROOT_DIR=/data1/home/luc.giffon/DeployedProjects/qalm_qmeans
PYTHON_EXEC=/data1/home/luc.giffon/anaconda3/envs/pyqalm/bin/python
PARAMETER_FILE=$ROOT_DIR/parameters/2020/10/8_9_lazyfile_kmeans_proj_l1_coil_kddcup09.txt
SCRIPT_FILE=$ROOT_DIR/code/scripts/2020/10/8_9_qmeans_l1_proj.py

nb_total=`wc -l ${PARAMETER_FILE} | cut -f1 -d' '`


SIZE_PARAMETER_FILE=$(wc -l $PARAMETER_FILE)

# if [ "$SLURM_ARRAY_TASK_ID" -gt "$SIZE_PARAMETER_FILE" ]; then
#    exit 0
# fi

# nettoyage des modules charges en interactif et herites par defaut
# module purge

# chargement des modules
# module load ...
# module load anaconda-py3/2019.03

# conda activate palmnet-gpu


LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p $PARAMETER_FILE)
echo $LINE

# echo des commandes lancees
set -x
# execution du code
srun $PYTHON_EXEC $SCRIPT_FILE $LINE