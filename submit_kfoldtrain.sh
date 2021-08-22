#!/bin/bash

#SBATCH --job-name=ensemble_RMdetection_5fold
#SBATCH --array=0-4
#SBATCH --time=0-05:59:00
##SBATCH --gres=gpu:p100:1
#SBATCH --output=%x--%A_%a.out
#SBATCH --mem=20GB
#SBATCH --account=*******

echo '******************Program Start*****************'
echo "Start time ="
date

mkdir -p weights

module load StdEnv/2018.3

module load python/3.7.4

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip -q

pip install tensorflow_gpu --no-index -q
echo '************************************************************************************'
pip install scikit-learn==0.23.0 --no-index -q
echo '************************************************************************************'
module load scipy-stack/2019a

pip install scikit-image --no-index -q
pip install six --no-index -q
pip install tqdm --no-index -q
pip install h5py --no-index -q
pip install opencv_python --no-index -q

pip install ~/projects/def-erangauk-ab/agarwala/ensemble/attrdict-2.0.1-py2.py3-none-any.whl -q
pip install ~/projects/def-erangauk-ab/agarwala/create_dataset/torchio-0.18.32-py2.py3-none-any.whl --no-index -q
pip install ~/projects/def-erangauk-ab/agarwala/ensemble/pynrrd-0.4.2-py2.py3-none-any.whl -q

## Prepare data
mkdir $SLURM_TMPDIR/data
tar xf ~/projects/def-erangauk-ab/agarwala/ensemble/ngmri_healthy_WITH_Bkgr.tar -C $SLURM_TMPDIR/data
tar xf ~/projects/def-erangauk-ab/agarwala/processed_data/dicom_dataset/GT_mri_kidney_segs.tar -C $SLURM_TMPDIR/data
tar xf ~/projects/def-erangauk-ab/agarwala/medphys/ensemble_tests/lesion_seg/GTkid_lesionSeg_results_$SLURM_ARRAY_TASK_ID.tar -C $SLURM_TMPDIR/data

echo '************************************************************************************'
echo "Data Extraction complete @"
date

## Note: array task ID tells the instance which fold it is working on
python train.py $SLURM_ARRAY_TASK_ID

echo '************************************************************************************'
echo "Finished Training @"
date

mkdir $SLURM_TMPDIR/results
mkdir $SLURM_TMPDIR/examples
python predict.py $SLURM_ARRAY_TASK_ID

## Transfer results and visualizations from the working node to the project node for viewing
cd $SLURM_TMPDIR
tar -cf $SLURM_TMPDIR/GTkid_2Chan_onlyHealthy_examples_$SLURM_ARRAY_TASK_ID.tar examples
cp $SLURM_TMPDIR/GTkid_2Chan_onlyHealthy_examples_$SLURM_ARRAY_TASK_ID.tar ~/projects/def-erangauk-ab/agarwala/medphys/ensemble_tests

cd $SLURM_TMPDIR
tar -cf $SLURM_TMPDIR/CRITICAL_verif_results_$SLURM_ARRAY_TASK_ID.tar results
cp $SLURM_TMPDIR/CRITICAL_verif_results_$SLURM_ARRAY_TASK_ID.tar ~/projects/def-erangauk-ab/agarwala/ensemble/ensemble_seg

echo '************************************************************************************'
echo "Time at exit @"
date