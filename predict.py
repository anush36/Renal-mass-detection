import json
import sys
from attrdict import AttrDict
import h5py
import numpy as np
import os
import pathlib
import sys
import tensorflow as tf
import glob
import time
import subprocess
from statistics import mean, stdev

import data_generator
import utils as utils
import pred_generator

i = int(sys.argv[1])

project_name = "fullcval_healthyNGMRI" + f"_{i}"
weight_path = "weights"

tmp_dir = subprocess.getoutput('echo $SLURM_TMPDIR')
origin = os.path.join(tmp_dir, 'data', 'ngmri_healthy_WITH_Bkgr')
og_dir = os.path.join(tmp_dir, 'data', 'GT_mri_kidney_segs')
lesion_dir = os.path.join(tmp_dir, 'data', 'results')
parameter_path = os.path.join(weight_path, project_name + ".json")

with open(parameter_path, "r") as file:
    PARAMS = json.load(file)

PARAMS = AttrDict(PARAMS)
utils.PARAMS = PARAMS
utils.TMPDIR = tmp_dir

model_weights = h5py.File(os.path.join(weight_path, project_name + ".hdf5"), 'r')

model = tf.keras.models.load_model(model_weights, custom_objects={'bce_fbeta_loss': utils.bce_fbeta_loss,
                                                'dice_loss': utils.dice_loss})

mri_dataset = data_generator.theDataset(origin=origin, PARAMS=PARAMS, balancing=False, TwoChan=True)
mri_dataset.full_dataset_k_fold_split(5)

predicter = pred_generator.predicter(PARAMS=PARAMS, model=model, dataset=mri_dataset, tmp_dir=tmp_dir, og_dir=og_dir, kidney=False, ensemble=lesion_dir)

predicter.predict_kfoldvalset(i, to_plot=True, save_predictions=False)

detection_precision = predicter.TP / (predicter.TP + predicter.FP)
detection_recall = predicter.TP / (predicter.TP + predicter.FN)

# Print all results and metrics out. 

"""
# Alternatively, print all results out to a file:

file1 = open(f"{project_name}.txt","w+")
file1.write(f"DICE: {np.mean(np.array(predicter.dice_list))} +- {np.std(np.array(predicter.dice_list))}\n")
file1.write(f"Seg Precision: {np.mean(np.array(predicter.precision_list))} +- {np.std(np.array(predicter.precision_list))}\n")
file1.write(f"Seg Recall: {np.mean(np.array(predicter.recall_list))} +- {np.std(np.array(predicter.recall_list))}\n")
file1.write(f"Avg Maximum Hausdorff Distance: {np.mean(np.array(predicter.hausdorff_list))} +- {np.std(np.array(predicter.hausdorff_list))}\n")
file1.write(f"Avg Absolute Volume Difference: {np.mean(np.array(predicter.abs_vol_dif_list))} +- {np.std(np.array(predicter.abs_vol_dif_list))}\n")
file1.write(f"Avg AUC: {np.mean(np.array(predicter.auc_list))} +- {np.std(np.array(predicter.auc_list))}\n")
file1.write(f"AVG Centroid Distance: {np.mean(np.array(predicter.centroid_list))} +- {np.std(np.array(predicter.centroid_list))}")
file1.write(f"Detection Precision: {detection_precision}")
file1.write(f"Detection Recall: {detection_recall}")
file1.close()
"""

print(mri_dataset.val_slices_in_PID[i][:,0])
print("\n-------------------------------------------------")

print(f"DICE: {predicter.dice_list}\n")
print(f"Seg Precision: {predicter.precision_list}\n")
print(f"Seg Recall: {predicter.recall_list}\n")
print(f"Avg Maximum Hausdorff Distance: {predicter.hausdorff_list}\n")
print(f"Avg Absolute Volume Difference: {predicter.abs_vol_dif_list}\n")
print(f"Avg AUC: {predicter.auc_list}\n")

print("\nLESION SEG RESULTS\n")
print(f"DICE: {predicter.lesion_dice_list}\n")
print(f"Seg Precision: {predicter.lesion_precision_list}\n")
print(f"Seg Recall: {predicter.lesion_recall_list}\n")
print(f"Avg Maximum Hausdorff Distance: {predicter.lesion_hausdorff_list}\n")
print(f"Avg Absolute Volume Difference: {predicter.lesion_abs_vol_dif_list}\n")

print(f"Centroid Distances: {predicter.centroid_list}")
print(f"True positives:{predicter.TP}")
print(f"False positives:{predicter.FP}")
print(f"False negatives:{predicter.FN}")

print("\n-------------------------------------------------")

print(f"DICE: {np.mean(np.array(predicter.dice_list))} +- {np.std(np.array(predicter.dice_list))}")
print(f"Seg Precision: {np.mean(np.array(predicter.precision_list))} +- {np.std(np.array(predicter.precision_list))}")
print(f"Seg Recall: {np.mean(np.array(predicter.recall_list))} +- {np.std(np.array(predicter.recall_list))}")
print(f"Avg Maximum Hausdorff Distance: {np.mean(np.array(predicter.hausdorff_list))} +- {np.std(np.array(predicter.hausdorff_list))}")
print(f"Avg Absolute Volume Difference: {np.mean(np.array(predicter.abs_vol_dif_list))} +- {np.std(np.array(predicter.abs_vol_dif_list))}")
print(f"Avg AUC: {np.mean(np.array(predicter.auc_list))} +- {np.std(np.array(predicter.auc_list))}")

print("\nLESION SEG RESULTS\n")
print(f"DICE: {np.mean(np.array(predicter.lesion_dice_list))} +- {np.std(np.array(predicter.lesion_dice_list))}")
print(f"Seg Precision: {np.mean(np.array(predicter.lesion_precision_list))} +- {np.std(np.array(predicter.lesion_precision_list))}")
print(f"Seg Recall: {np.mean(np.array(predicter.lesion_recall_list))} +- {np.std(np.array(predicter.lesion_recall_list))}")
print(f"Avg Maximum Hausdorff Distance: {np.mean(np.array(predicter.lesion_hausdorff_list))} +- {np.std(np.array(predicter.lesion_hausdorff_list))}")
print(f"Avg Absolute Volume Difference: {np.mean(np.array(predicter.lesion_abs_vol_dif_list))} +- {np.std(np.array(predicter.lesion_abs_vol_dif_list))}")

print(f"AVG Centroid Distance: {np.mean(np.array(predicter.centroid_list))} +- {np.std(np.array(predicter.centroid_list))}")
print(f"Detection Precision: {detection_precision}")
print(f"Detection Recall: {detection_recall}")

print("-------------------------------------------------\n")


