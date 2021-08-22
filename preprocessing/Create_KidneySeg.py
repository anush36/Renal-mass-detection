import glob
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchio as tio
import subprocess
import random
from skimage import exposure
import warnings

# Visualize some examples of pre-processed data
def save_examples(img, seg, PID, fname):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.subplot(1, 2, 2)
    plt.imshow(seg)
    plt.title("Segmentation")
    plt.suptitle('PID: {}, Filename: {}'.format(PID, fname))
    fig.savefig(os.path.join(example_dir, f'{PID}_{fname}.png'))

# Creates filepaths for saving pre-processed data, applies histogram equalization, and saves axial slices of MRI volume
def image_saver(the_image, the_seg, orientation, pid, save_directory):

    img_save_path = os.path.join(save_directory, pid, 'images')
    Path(img_save_path).mkdir(exist_ok=True, parents=True)

    seg_save_path = os.path.join(save_directory, pid, 'seg')
    Path(seg_save_path).mkdir(exist_ok=True, parents=True)

    for x in range(the_image.shape[0]):
        a_img = the_image[x, :, :]
        a_seg = the_seg[x, :, :]

        try:
            a_img = exposure.equalize_adapthist(a_img)
        except:
            print(f"EXCEPTION TRIGGERED: {pid}")
            a_img = exposure.equalize_adapthist(a_img.astype(np.int32))

        if x - 100 < 0:
            slice_index = '0' + str(x)
            if x - 10 < 0:
                slice_index = '0' + slice_index
        else:
            slice_index = x

        fname = "{}.npy".format(slice_index)
        
        np.save(os.path.join(img_save_path, orientation + fname), a_img, allow_pickle=False)
        np.save(os.path.join(seg_save_path, orientation + fname), a_seg, allow_pickle=False)

        if random.random() < 0.001:
            save_examples(a_img, a_seg, pid, orientation + fname)

tmp_dir = subprocess.getoutput('echo $SLURM_TMPDIR')

paths = glob.glob(os.path.join(tmp_dir, 'data', 'RenalMassMRIDICOM_NEW_DICOM_Images', '*'))
seg_dir = os.path.join(tmp_dir, 'data', 'GT_mri_kidney_segs')

example_dir = os.path.join(tmp_dir, 'examples')

# set target voxel size. (1, 1, 1) creates an isotropic voxel
transform = tio.Resample((1.0, 1.0, 1.0))

#pre-defined train/val/test split by patient id (PID). Not used for k-fold cross validation
test_list = [3, 96, 48, 111, 116, 13, 36, 70, 50, 6, 58, 28]
val_list = [88, 53, 66, 74, 62, 5, 45, 90, 117, 101, 108, 19]

for path in tqdm(paths, mininterval=10):

    locs = path.split(os.path.sep)

    if int(locs[-1]) in test_list:
        continue
        save_dir = os.path.join(tmp_dir, 'results', 'ngmri_wholeKidney_adahist_voxnorm', 'test')
    elif int(locs[-1]) in val_list:
        continue
        save_dir = os.path.join(tmp_dir, 'results', 'ngmri_wholeKidney_adahist_voxnorm', 'val')
    else:
        save_dir = os.path.join(tmp_dir, 'results', 'ngmri_wholeKidney_adahist_voxnorm', 'train')
    
    img_path = glob.glob(os.path.join(path, '*NG*'))

    # Load original DICOM volume for a given patient and normalize voxel size
    t_img = tio.ScalarImage(img_path)
    t_img = transform(t_img)
    seg_transform = tio.Resample(t_img)
    img = t_img.numpy()
    if img.shape[0] != 1:
        img = img[0, :, :, :]
    img = np.squeeze(img)

    # Load original (ground truth) segmentation data
    seg_path = glob.glob(os.path.join(seg_dir, locs[-1], '*NG*'))
    t_seg = tio.LabelMap(seg_path)
    t_seg = seg_transform(t_seg)
    seg = t_seg.numpy()
    seg = np.squeeze(seg)

    # Convert ground truth segmentation to binary, where all labels not in background are made foreground
    seg = np.where(seg != 0, 1, 0)

    # Rotate image planes to ensure slices are saved in axial view (clinically standardized view)
    if locs[-1] == '66':
        img = np.rot90(img, 1, (0,1))
        img = np.rot90(img, 1, (2,1))
        seg = np.rot90(seg, 1, (0,1))
        seg = np.rot90(seg, 1, (2,1))
    
    else:
        seg = np.rot90(seg, 1, (0,2))
        img = np.rot90(img, 1, (0,2))

    image_saver(img, seg, "", locs[-1], save_dir)



