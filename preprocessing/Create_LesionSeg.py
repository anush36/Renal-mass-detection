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
import tensorflow as tf

warnings.filterwarnings("ignore")

i = int(sys.argv[1])

# Resize predicted kidney boundary to original DICOM size for kidney boundary extraction
def tf_resize(x, max_size, target_x, target_y):

    full_volume = x[0, :, :]
    full_volume = np.expand_dims(full_volume, axis=-1)
    full_volume = tf.convert_to_tensor(full_volume)
    full_volume = tf.image.resize(full_volume, [max_size, max_size], method='nearest')
    full_volume = tf.image.resize_with_crop_or_pad(full_volume, target_x, target_y)
    full_volume = full_volume.numpy()
    full_volume = np.squeeze(full_volume)
    full_volume = np.expand_dims(full_volume, axis=0)

    for y in range(1, x.shape[0]):
        aslice = x[y, :, :]
        aslice = np.expand_dims(aslice, axis=-1)
        aslice = tf.convert_to_tensor(aslice)
        aslice = tf.image.resize(aslice, [max_size, max_size], method='nearest')
        aslice = tf.image.resize_with_crop_or_pad(aslice, target_x, target_y)
        aslice = aslice.numpy()
        aslice = np.squeeze(aslice)
        aslice = np.expand_dims(aslice, axis=0)
        full_volume = np.append(full_volume, aslice, axis=0)
    return full_volume

# Visualize some examples of pre-processed data
def save_examples(img, seg, PID, fname):
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img[:, :, 0], cmap='gray')
    plt.title("Image_channel")
    plt.subplot(1, 3, 2)
    plt.imshow(img[:, :, 1], cmap='gray')
    plt.title("Kidney_channel")
    plt.subplot(1, 3, 3)
    plt.imshow(seg, cmap='gray')
    plt.title("healthy_seg")
    plt.suptitle('PID: {}, Filename: {}'.format(PID, fname))
    fig.savefig(os.path.join(example_dir, f'{PID}_{fname}.png'))

# Creates filepaths for saving pre-processed data and saves axial slices of MRI volume
def image_saver(the_image, the_seg, orientation, pid, save_directory):

    for x in range(the_image.shape[0]):
        a_img = the_image[x, :, :, :]
        a_seg = the_seg[x, :, :]

        # save image slices based on their class, useful for over-sampling under-represented classes during training
        if np.any(a_seg == 1):
            img_save_path = os.path.join(save_directory, pid, 'positive', 'images')
            Path(img_save_path).mkdir(exist_ok=True, parents=True)
            seg_save_path = os.path.join(save_directory, pid, 'positive', 'seg')
            Path(seg_save_path).mkdir(exist_ok=True, parents=True)
        else:
            img_save_path = os.path.join(save_directory, pid, 'negative', 'images')
            Path(img_save_path).mkdir(exist_ok=True, parents=True)
            seg_save_path = os.path.join(save_directory, pid, 'negative', 'seg')
            Path(seg_save_path).mkdir(exist_ok=True, parents=True)
        
        # also save a set of the images in the original order, useful for model evaluation where patient slices must be presented in order
        neutral_img_save_path = os.path.join(save_directory, pid, 'neutral', 'images')
        Path(neutral_img_save_path).mkdir(exist_ok=True, parents=True)
        neutral_seg_save_path = os.path.join(save_directory, pid, 'neutral', 'seg')
        Path(neutral_seg_save_path).mkdir(exist_ok=True, parents=True)
        
        if x - 100 < 0:
            slice_index = '0' + str(x)
            if x - 10 < 0:
                slice_index = '0' + slice_index
        else:
            slice_index = x

        fname = "{}.npy".format(slice_index)
        
        np.save(os.path.join(img_save_path, orientation + fname), a_img, allow_pickle=False)
        np.save(os.path.join(seg_save_path, orientation + fname), a_seg, allow_pickle=False)
        
        np.save(os.path.join(neutral_img_save_path, orientation + fname), a_img, allow_pickle=False)
        np.save(os.path.join(neutral_seg_save_path, orientation + fname), a_seg, allow_pickle=False)

        if random.random() < 1:
            save_examples(a_img, a_seg, pid, orientation + fname)

tmp_dir = subprocess.getoutput('echo $SLURM_TMPDIR')

# Load DICOM volumes, ground truth segmentations, and predicted kidney segmentations
paths = glob.glob(os.path.join(tmp_dir, 'data', f'final_kidseg_wholeNGMRI_results_{i}', '*'))
og_img_dir = os.path.join(tmp_dir, 'data', 'RenalMassMRIDICOM_NEW_DICOM_Images')
seg_dir = os.path.join(tmp_dir, 'data', 'GT_mri_kidney_segs')

# Directory to store examples of pre-processed data
example_dir = os.path.join(tmp_dir, 'examples')

# set target voxel size. (1, 1, 1) creates an isotropic voxel
transform = tio.Resample((1.0, 1.0, 1.0))

for path in tqdm(paths, mininterval=10):

    locs = path.split(os.path.sep)

    save_dir = os.path.join(tmp_dir, 'results', f'predKid_TwoChan_healthySeg_{i}', 'test')

    # load predicted kidney segmentations for a given patient
    pred_seg = np.load(os.path.join(path, 'pred_segs', '000.npy'))
    pred_seg = np.expand_dims(pred_seg, axis=0)
    for z in range(1, len(glob.glob(os.path.join(path, 'pred_segs', '*')))):
    
        if z - 100 < 0:
            slice_name = '0' + str(z)
            if z - 10 < 0:
                slice_name = '0' + slice_name
        else:
            slice_name = z
        slice_name = "{}.npy".format(slice_name)

        a_pred_slice = np.load(os.path.join(path, 'pred_segs', slice_name))
        a_pred_slice = np.expand_dims(a_pred_slice, axis=0)
        
        pred_seg = np.concatenate((pred_seg, a_pred_slice), axis=0)

    # Load original DICOM volume for a given patient and normalize voxel size
    img_path = glob.glob(os.path.join(og_img_dir, locs[-1], '*NG*'))
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
    
    if locs[-1] =='66':
        img = np.rot90(img, 1, (0,1))
        img = np.rot90(img, 1, (2,1))
        seg = np.rot90(seg, 1, (0,1))
        seg = np.rot90(seg, 1, (2,1))
    else:
        seg = np.rot90(seg, 1, (0,2))
        img = np.rot90(img, 1, (0,2))

    max_size = max([img.shape[1], img.shape[2]])
    pred_seg = tf_resize(pred_seg, max_size, img.shape[1], img.shape[2])

    pred_foreground = np.where(np.greater_equal(pred_seg, 0.5), 1, 0)
    
    # Convert segmentations to binary labels, with solid tumors and cysts all belonging to the foreground
    lesion_seg = np.where(np.logical_or(np.logical_and(seg >= 1, seg <= 3), seg == 5), 1, 0)
    foreground_seg = np.where(seg != 0, 1, 0)

    # Split images along the saggital plane to separate left and right kidney
    length = np.floor_divide(seg.shape[2], 2)
    R_pred_foreground = pred_foreground[:, :, length:]
    R_foreground = foreground_seg[:, :, length:]
    R_seg = lesion_seg[:, :, length:]
    R_img = img[:, :, length:]
    R_xs, R_ys, R_zs = np.where(R_pred_foreground != 0)
    
    # Isolate split volumes to region of interest containing kidney, apply histogram equalization, and save pre-processed images through image_saver()
    
    if all([R_xs.shape[0], R_ys.shape[0], R_zs.shape[0]]):
        R_seg = R_seg[min(R_xs):max(R_xs), min(R_ys):max(R_ys), min(R_zs):max(R_zs)]
        R_img = R_img[min(R_xs):max(R_xs), min(R_ys):max(R_ys), min(R_zs):max(R_zs)]
        R_foreground = R_foreground[min(R_xs):max(R_xs), min(R_ys):max(R_ys), min(R_zs):max(R_zs)]
        R_pred_foreground = R_pred_foreground[min(R_xs):max(R_xs), min(R_ys):max(R_ys), min(R_zs):max(R_zs)]

        try:
            R_img = exposure.equalize_adapthist(R_img)
        except:
            R_img = exposure.equalize_adapthist(R_img.astype(np.int32))
            
        R_img = np.expand_dims(R_img, axis = -1)
        R_pred_foreground = np.expand_dims(R_pred_foreground, axis = -1)
        R_img = np.concatenate((R_img, R_pred_foreground), axis = -1)    
    
        image_saver(R_img, R_seg, "L", locs[-1], save_dir)

    L_pred_foreground = pred_foreground[:, :, 0:length]
    L_foreground = foreground_seg[:, :, 0:length]
    L_seg = lesion_seg[:, :, 0:length]
    L_img = img[:, :, 0:length]
    L_xs, L_ys, L_zs = np.where(L_pred_foreground != 0)
    
    if all([L_xs.shape[0], L_ys.shape[0], L_zs.shape[0]]):
        L_seg = L_seg[min(L_xs):max(L_xs), min(L_ys):max(L_ys), min(L_zs):max(L_zs)]
        L_img = L_img[min(L_xs):max(L_xs), min(L_ys):max(L_ys), min(L_zs):max(L_zs)]
        L_foreground = L_foreground[min(L_xs):max(L_xs), min(L_ys):max(L_ys), min(L_zs):max(L_zs)]
        L_pred_foreground = L_pred_foreground[min(L_xs):max(L_xs), min(L_ys):max(L_ys), min(L_zs):max(L_zs)]

        try:
            L_img = exposure.equalize_adapthist(L_img)
        except:
            L_img = exposure.equalize_adapthist(L_img.astype(np.int32))
        
        L_img = np.expand_dims(L_img, axis = -1)
        L_pred_foreground = np.expand_dims(L_pred_foreground, axis = -1)
        L_img = np.concatenate((L_img, L_pred_foreground), axis = -1)
            
        image_saver(L_img, L_seg, "R", locs[-1], save_dir)
  