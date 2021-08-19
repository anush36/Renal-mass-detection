import tensorflow as tf
from tensorflow.python.keras import losses
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from scipy.ndimage import map_coordinates, gaussian_filter #used for elastic deformation
import skimage.metrics
import skimage.segmentation
import skimage.feature
import skimage.measure
from scipy.spatial import cKDTree
import sys
import os
import glob
import torchio as tio
import cv2
import nrrd  #pip install ~/projects/def-erangauk-ab/agarwala/Final/pynrrd-0.4.2-py2.py3-none-any.whl  -q
import warnings
import subprocess


def tf_resize(x):

    full_volume = x[0, :, :]
    full_volume = np.expand_dims(full_volume, axis=-1)
    full_volume = tf.convert_to_tensor(full_volume)
    asize = tf.math.reduce_max(tf.shape(full_volume))
    full_volume = tf.image.resize_with_crop_or_pad(full_volume, asize, asize)
    full_volume = tf.image.resize(full_volume, [PARAMS.IMG_SIZE, PARAMS.IMG_SIZE], method='nearest')
    full_volume = full_volume.numpy()
    full_volume = np.squeeze(full_volume)
    full_volume = np.expand_dims(full_volume, axis=0)

    for y in range(1, x.shape[0]):
        aslice = x[y, :, :]
        aslice = np.expand_dims(aslice, axis=-1)
        aslice = tf.convert_to_tensor(aslice)
        asize = tf.math.reduce_max(tf.shape(aslice))
        aslice = tf.image.resize_with_crop_or_pad(aslice, asize, asize)
        aslice = tf.image.resize(aslice, [PARAMS.IMG_SIZE, PARAMS.IMG_SIZE], method='nearest')
        aslice = aslice.numpy()
        aslice = np.squeeze(aslice)
        aslice = np.expand_dims(aslice, axis=0)
        full_volume = np.append(full_volume, aslice, axis=0)

    return full_volume


def blob_detector(img):
    img = img.astype(np.uint8)

    blobs = skimage.feature.blob_log(img*100, overlap=0, min_sigma=2)

    if blobs.shape[0] == 0:
        num_masses = 0
        no_blob = np.zeros(1)
        return no_blob, num_masses

    top_logs = np.flip(np.argsort(blobs[:,3]))

    num_masses = 0

    good_mask = skimage.segmentation.flood(img, (int(blobs[top_logs[0], 0]), int(blobs[top_logs[0], 1]), int(blobs[top_logs[0], 2])))
    good_blobs = np.array([int(blobs[top_logs[0], 0]), int(blobs[top_logs[0], 1]), int(blobs[top_logs[0], 2])])
    good_blobs = np.expand_dims(good_blobs, axis=0)
    num_masses += 1

    for x in range(1, blobs.shape[0]):
        test_mask = skimage.segmentation.flood(img, (int(blobs[top_logs[x], 0]), int(blobs[top_logs[x], 1]), int(blobs[top_logs[x], 2])))
        if np.any(np.multiply(good_mask, test_mask)):
            continue
        else:
            good_mask = np.where(np.logical_or(good_mask == True, test_mask == True), True, False)
            blob_to_add = np.array([int(blobs[top_logs[x], 0]), int(blobs[top_logs[x], 1]), int(blobs[top_logs[x], 2])])
            blob_to_add = np.expand_dims(blob_to_add, axis=0)
            good_blobs = np.append(good_blobs, blob_to_add, axis=0)
            num_masses += 1

    return good_blobs, num_masses

def highlight_blobs(pred, pid, name, example_dir):

    no_blob = np.zeros(1)
    
    #print(blobs.shape, "blobs in prediction")

    #output_path = os.path.join('C:/', 'Users', 'Anush Agarwal', 'Documents', 'kidney_preds', '2021mar', pid)
    #Path(output_path).mkdir(exist_ok=True, parents = True)

    blobs, _ = blob_detector(pred)
    
    if blobs.shape != no_blob.shape:
        good_mask = skimage.segmentation.flood(pred, (int(blobs[0, 0]), int(blobs[0, 1]), int(blobs[0, 2])))
        ras_pos = np.where(good_mask==True, 100, 0)

        for x in range(1, blobs.shape[0]):
            good_mask = skimage.segmentation.flood(pred, (int(blobs[x, 0]), int(blobs[x, 1]), int(blobs[x, 2])))
            add_ras_pos = np.where(good_mask==True, (x+1)*100, 0)
            ras_pos = ras_pos + add_ras_pos

        #img_output = os.path.join('C:/', 'Users', 'Anush Agarwal', 'Documents', 'kidney_preds', '2021mar', 'kidney_img_forBlob.nrrd')

        #utils.visualize(img, ras_pos, ras_pos)

        nrrd.write(os.path.join(example_dir, f'{pid}_{name}_blobDet.nrrd'), ras_pos)
    
    else:
        print(f"HIGHLIGHT BLOBs: {pid}_{name} no blobs to highlight")
    

def compare_blobs(kidscreen, healthy_pred, gt, pred, PID, og_dir=None, Ensemble=False):


    a_cent_list = []
    
    TP = 0
    FP = 0
    FN = 0
    no_blob = np.zeros(1)
    
    # GET THE GROUND TRUTH DIRECTLY FROM SOURCE. This opens up possibility of getting tumor/cyst specific results etc

    transform = tio.Resample((1.00, 1.00, 1.00))

    #img_dir = glob.glob(os.path.join(og_img_dir, PID, '*NG*'))
    #t_img = tio.ScalarImage(og_img_path)
    #fixed_img = transform(t_img)
    #t_resamp = tio.Resample(fixed_img)
    #show_img = np.squeeze(fixed_img.numpy())
    #show_img = np.rot90(show_img, 1, (0,2))
    
    if og_dir != None:

        og_gt_path = os.path.join(og_dir, PID, '*NG*')
        print(og_gt_path)
        og_gt_path = glob.glob(og_gt_path)
        t_seg = tio.LabelMap(og_gt_path)
        try:
            fixed_seg = transform(t_seg)
        except:
            print(og_dir)
            print(PID)
            print(og_gt_path, "\nError Transforming Ground Truth Labels")
            sys.exit(0)
        the_seg = np.squeeze(fixed_seg.numpy())

        if PID != '66':
            the_seg = np.rot90(the_seg, 1, (0,2))
        else:
            the_seg = np.rot90(the_seg, 1, (0,1))
            the_seg = np.rot90(the_seg, 1, (2,1))

        #the_seg = tf_resize(the_seg)
        the_foreground = np.where(the_seg != 0, 1, 0)
        #the_seg = np.where(the_seg != 0, 1, 0)
        #the_seg = np.where(np.logical_and(the_seg >=1, the_seg <= 3), 1, 0)
        the_seg = np.where(np.logical_or(np.logical_and(the_seg >=1, the_seg <= 3), the_seg == 5), 1, 0)
        #the_seg = np.where(np.logical_or(the_seg == 4, the_seg == 6), 1, 0)
        length = np.floor_divide(the_foreground.shape[2], 2)

        L_foreground = the_foreground[:, :, 0:length]
        R_foreground = the_foreground[:, :, length:]
        
        L_gt = the_seg[:, :, 0:length]
        R_gt = the_seg[:, :, length:]

        L_xs, L_ys, L_zs = np.where(L_foreground != 0)
        R_xs, R_ys, R_zs = np.where(R_foreground != 0)

        if all([R_xs.shape[0], R_ys.shape[0], R_zs.shape[0]]) and all([L_xs.shape[0], L_ys.shape[0], L_zs.shape[0]]):
            L_gt_seg = R_gt[min(R_xs):max(R_xs), min(R_ys):max(R_ys), min(R_zs):max(R_zs)]
            L_gt_seg = tf_resize(L_gt_seg)
            R_gt_seg = L_gt[min(L_xs):max(L_xs), min(L_ys):max(L_ys), min(L_zs):max(L_zs)]
            R_gt_seg = tf_resize(R_gt_seg)
            the_gt = np.concatenate((L_gt_seg, R_gt_seg), axis=0)
            
            another_L_foreground = tf_resize(R_foreground[min(R_xs):max(R_xs), min(R_ys):max(R_ys), min(R_zs):max(R_zs)])
            another_R_foreground = tf_resize(L_foreground[min(L_xs):max(L_xs), min(L_ys):max(L_ys), min(L_zs):max(L_zs)])
            the_foreground = np.concatenate((another_L_foreground, another_R_foreground), axis=0)

        elif all([R_xs.shape[0], R_ys.shape[0], R_zs.shape[0]]):
            L_gt_seg = R_gt[min(R_xs):max(R_xs), min(R_ys):max(R_ys), min(R_zs):max(R_zs)]
            the_gt = tf_resize(L_gt_seg)
            
            the_foreground = tf_resize(R_foreground[min(R_xs):max(R_xs), min(R_ys):max(R_ys), min(R_zs):max(R_zs)])

        else:
            R_gt_seg = L_gt[min(L_xs):max(L_xs), min(L_ys):max(L_ys), min(L_zs):max(L_zs)]
            the_gt = tf_resize(R_gt_seg)
            
            the_foreground = tf_resize(L_foreground[min(L_xs):max(L_xs), min(L_ys):max(L_ys), min(L_zs):max(L_zs)])

        print(f"For case {PID}: gt:{the_gt.shape} pred:{pred.shape}")    

    else:
        # PROVIDE THE GT DIRECTLY FROM DATA PIPELINE
        the_gt = gt

    
    if Ensemble:
        
        # MEAN AVERAGING - Proposed Method
        #convert healthy pred to indirect renal mass segmentation
        healthy_pred = np.subtract(kidscreen, healthy_pred)
        #perform mean averaging ensemble
        interim_pred = np.mean(np.array([pred, healthy_pred]), axis=0)
        #threshold the final ensemble output
        pred = np.where(np.greater_equal(interim_pred, 0.5), 1, 0)


        # ALTERNATIVE METHODS - For Reference Only

        """
        # Only using indirect renal mass segmentation
        pred = np.subtract(the_foreground, healthy_pred)
        pred = np.where(np.greater_equal(pred, 0.5), 1, 0)
        """

        """
        # Healthy Gating
        healthy_pred = np.where(np.greater_equal(healthy_pred, 0.5), 1, 0)
        pred = np.where(np.greater_equal(pred, 0.5), 1, 0)
        pred = np.where(np.logical_and(healthy_pred != 1, pred == 1), 1, 0)
        """


    gt_blobs, _ = blob_detector(the_gt)
    pred_blobs, _ = blob_detector(pred)
    
    # trivial scenarios
    if gt_blobs.shape == no_blob.shape and pred_blobs.shape == no_blob.shape:
        print("No blobs in either")
        print("**************NO GT BLOB FOUND IN THIS CASE AT ALL!*************")
        pred = np.where(pred > 0.5, 1, 0)
        return TP, FP, FN, a_cent_list, the_gt, pred

    if gt_blobs.shape == no_blob.shape and pred_blobs.shape != no_blob.shape:
        print("All false positives")
        print("**************NO GT BLOB FOUND IN THIS CASE AT ALL!*************")
        FP = pred_blobs.shape[0]
        pred = np.where(pred > 0.5, 1, 0)
        return TP, FP, FN, a_cent_list, the_gt, pred

    if gt_blobs.shape != no_blob.shape and pred_blobs.shape == no_blob.shape:
        #print("All false negatives")
        FN = gt_blobs.shape[0]
        pred = np.where(pred > 0.5, 1, 0)
        return TP, FP, FN, a_cent_list, the_gt, pred

    # COMPARE BLOBS

    # create union of all TP blobs and find TP and FN
    matching_blobs = np.full((the_gt.shape[0], the_gt.shape[1], the_gt.shape[2]), False, dtype=np.bool_)

    for x in range(gt_blobs.shape[0]):
        gt_test_mask = skimage.segmentation.flood(gt, (gt_blobs[x, 0], gt_blobs[x, 1], gt_blobs[x, 2]))
        print(f"Ground Truth Centroid for PID {PID}: {[gt_blobs[x, 0], gt_blobs[x, 1], gt_blobs[x, 2]]}")
        
        initial_TP = TP

        # check the gt blob against each pred blob, if there is a match, treat as true positive
        for y in range(pred_blobs.shape[0]):
            pred_test_mask = skimage.segmentation.flood(pred, (pred_blobs[y, 0], pred_blobs[y, 1], pred_blobs[y, 2]))
            if np.any(np.multiply(gt_test_mask, pred_test_mask)):
                TP += 1
                print(f"Centroid for PID {PID}: {[pred_blobs[y, 0], pred_blobs[y, 1], pred_blobs[y, 2]]}")
                a = np.array([gt_blobs[x, 0], gt_blobs[x, 1], gt_blobs[x, 2]])
                b = np.array([pred_blobs[y, 0], pred_blobs[y, 1], pred_blobs[y, 2]])
                a_cent_list.append(abs(np.linalg.norm(a-b)))
                matching_blobs = np.where(np.logical_or(gt_test_mask == True, matching_blobs == True), True, False)
                break

        # if the gt blob is not a true positive match, it must be a false negative
        if TP == initial_TP:
            FN += 1

    # check for false positives by seeing if any pred_blobs do not belong to matching_blobs
    for x in range(pred_blobs.shape[0]):
        pred_test_mask = skimage.segmentation.flood(pred, (pred_blobs[x, 0], pred_blobs[x, 1], pred_blobs[x, 2]))

        if np.any(np.multiply(matching_blobs, pred_test_mask)):
            continue
        else:
            FP += 1

    #print(f"Patient: {PID} had {TP} TP, {FP} FP, {FN} FN")
    
    pred = np.where(pred > 0.5, 1, 0)
    
    return TP, FP, FN, a_cent_list, the_gt, pred


def postprocess_kidseg(seg, img=None):

    blobs, _ = blob_detector(seg)

    if blobs.ndim < 2:
        if img == None:
            return seg
        else:
            return seg, img
    
    first_kidney = skimage.segmentation.flood(seg, (int(blobs[0, 0]), int(blobs[0, 1]), int(blobs[0, 2])))
    
    if blobs.shape[0] == 1:
        if img == None:
            return np.where(first_kidney, seg, 0)
        else:
            return np.where(first_kidney, seg, 0), np.where(first_kidney, img, 0)
    
    second_kidney = skimage.segmentation.flood(seg, (int(blobs[1, 0]), int(blobs[1, 1]), int(blobs[1, 2])))
    cleaned_seg = np.where(np.logical_or(first_kidney, second_kidney), seg, 0)

    if img != None:
        if img.shape != seg.shape:
            sys.exit("Error: img and seg provided to postprocess_kidseg are not the same shape")
        cleaned_img = np.where(cleaned_seg != 0, img, 0)
        return cleaned_img, cleaned_seg
    else:
        return cleaned_seg


def max_hausdorff(pred, gt):

    # Find Hausdorff Distance error

    gt_hd = np.where(gt != 0, True, False)
    pred_hd = np.where(pred != 0, True, False)

    hd = skimage.metrics.hausdorff_distance(gt_hd, pred_hd)

    return hd

def hd_pairs(pred, gt):

    # Finds the co-ordinates of the predicted mask that have the largest Hausdorff Distance Error
    # Adapted from skimage.metrics.hausdorff_pair: https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.hausdorff_pair
    
    a_points = np.transpose(np.nonzero(pred))
    b_points = np.transpose(np.nonzero(gt))

    # If either of the sets are empty, there is no corresponding pair of points
    if len(a_points) == 0 or len(b_points) == 0:
        warnings.warn("One or both of the images is empty.", stacklevel=2)
        return (), ()

    nearest_dists_from_b, nearest_a_point_indices_from_b = cKDTree(a_points) \
        .query(b_points)
    nearest_dists_from_a, nearest_b_point_indices_from_a = cKDTree(b_points) \
        .query(a_points)

    max_index_from_a = nearest_dists_from_b.argmax()
    max_index_from_b = nearest_dists_from_a.argmax()

    max_dist_from_a = nearest_dists_from_b[max_index_from_a]
    max_dist_from_b = nearest_dists_from_a[max_index_from_b]

    if max_dist_from_b > max_dist_from_a:
        return a_points[max_index_from_b], \
            b_points[nearest_b_point_indices_from_a[max_index_from_b]]
    else:
        return a_points[nearest_a_point_indices_from_b[max_index_from_a]], \
            b_points[max_index_from_a]

def preThres_auc(pred, gt):

    # Find area under the curve metric of pre-thresholded segmentation

    gt_auc = gt.flatten()
    pred_auc = pred.flatten()
    
    if np.any(gt_auc == 1) and np.any(pred_auc == 1):
        auc = metrics.roc_auc_score(gt_auc, pred_auc)
    else:
        auc = 1

    return auc


def abs_vol_dif(pred, gt):

    # Find absolute volume difference error
    
    gt_values, gt_counts = np.unique(gt.astype(np.uint8), return_counts=True)
    
    if np.any(gt_values) == 1:
        gt_vol = gt_counts[gt_values==1][0]
    else:
        gt_vol = 0
        return 0

    pred_values, pred_counts = np.unique(pred.astype(np.uint8), return_counts=True)
    
    if np.any(pred_values) ==  1:
        pred_vol = pred_counts[pred_values==1][0]
    else:
        pred_vol = 0
    

    vol_dif = (pred_vol / gt_vol) - 1

    return abs(vol_dif)


def deform_grid (X, Y=None, Z=None, points=3):
    """
    TAKEN FROM: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py

    Elastic deformation of 2D or 3D images on a gridwise basis
    X: image
    Y: segmentation of the image
    sigma = standard deviation of the normal distribution
    points = number of points of the each side of the square grid
    Elastic deformation approach found in
        Ronneberger, Fischer, and Brox, "U-Net: Convolutional Networks for Biomedical
        Image Segmentation" also used in Çiçek et al., "3D U-Net: Learning Dense Volumetric
        Segmentation from Sparse Annotation"
    based on a coarsed displacement grid interpolated to generate displacement for every pixel
    deemed to represent more realistic, biologically explainable deformation of the image
    for each dimension, a value for the displacement is generated on each point of the grid
    then interpolated to give an array of displacement values, which is then added to the corresponding array of coordinates
    the resulting (list of) array of coordinates is mapped to the original image to give the final image
    """
    sigma=PARAMS.ELASTIC_SIGMA
    shape = X.shape
    if len(shape)==2:
        coordinates = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij') #creates the grid of coordinates of the points of the image (an ndim array per dimension)
        xi=np.meshgrid(np.linspace(0,points-1,shape[0]), np.linspace(0,points-1,shape[1]), indexing='ij') #creates the grid of coordinates of the points of the image in the "deformation grid" frame of reference
        grid = [points, points]
        
    elif len(shape)==3:
        coordinates = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij') #creates the grid of coordinates of the points of the image (an ndim array per dimension)
        xi = np.meshgrid(np.linspace(0,points-1,shape[0]), np.linspace(0,points-1,shape[1]), np.linspace(0,points-1,shape[2]), indexing='ij') #creates the grid of coordinates of the points of the image in the "deformation grid" frame of reference
        grid = [points, points, points]
        
    else:
        raise ValueError("can't deform because the image is not either 2D or 3D")

    for i in range(len(shape)): #creates the deformation along each dimension and then add it to the coordinates
        yi=np.random.randn(*grid)*sigma #creating the displacement at the control points
        y = map_coordinates(yi, xi, order=3).reshape(shape)
        #print(y.shape,coordinates[i].shape) #y and coordinates[i] should be of the same shape otherwise the same displacement is applied to every ?row? of points ?
        coordinates[i]=np.add(coordinates[i],y) #adding the displacement
    
    if Z is not None and Y is None:
        return map_coordinates(X, coordinates, order=3).reshape(shape), map_coordinates(Z, coordinates, order=3).reshape(shape)
    elif Z is not None and Y is not None:
        return map_coordinates(X, coordinates, order=3).reshape(shape), map_coordinates(Z, coordinates, order=3).reshape(shape), map_coordinates(Y, coordinates, order=0).reshape(shape)
    elif Y is None:
        return map_coordinates(X, coordinates, order=3).reshape(shape), None
    else:
        return map_coordinates(X, coordinates, order=3).reshape(shape), map_coordinates(Y, coordinates, order=0).reshape(shape)


def fbeta(y_true, y_pred):

    # Asymmetric similarity loss implementation

    if isinstance(y_true, np.ndarray):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)

    if isinstance(y_pred, np.ndarray):
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum(y_pred_f * (1 - y_true_f))
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    smooth = 1.
    beta = PARAMS.FBETA

    """
    beta coefficients:
    beta = 1 --> Dice coefficient
    beta = 2 --> F2 score
    beta = 0 --> precision only
    """

    fn_coeff = (beta**2) / (1 + beta**2)
    fp_coeff = 1 / (1 + beta**2)

    v2_score = (tp + smooth) / (tp + fn_coeff*fn + fp_coeff*fp + smooth)

    return v2_score


def fbeta_loss(y_true, y_pred):
    loss = (1 - fbeta(y_true, y_pred))
    return loss


def bce_fbeta_loss(y_true, y_pred):
    #loss = losses.binary_crossentropy(y_true, y_pred) + fbeta_loss(y_true, y_pred)
    loss = losses.binary_crossentropy(y_true, y_pred) + fbeta_loss(y_true, y_pred)
    return loss


def dice_coeff(y_true, y_pred):

    if isinstance(y_true, np.ndarray):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)

    if isinstance(y_pred, np.ndarray):
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred)
    return loss
    

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def visualize(x, y, truth, PID='x', DICE='x'):

    # Ensures that the files are numpy arrays, not tensors
    try:
        x = x.numpy()
    except:
        pass
    try:
        y = y.numpy()
    except:
        pass
    try:
        truth = truth.numpy()
    except:
        pass

    x = np.squeeze(x)
    y = np.squeeze(y)
    truth = np.squeeze(truth)

    if x.ndim == 3:
        for slice in range(x.shape[0]):
            fig = plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(x[slice, :, :])
            plt.title("Input")
            plt.subplot(1, 3, 2)
            plt.imshow(y[slice, :, :])
            plt.title("Prediction")
            plt.subplot(1, 3, 3)
            plt.imshow(truth[slice, :, :])
            plt.title("Ground Truth")
            plt.suptitle('Patient: {}, DICE: {}'.format(PID, DICE))
            plt.show()
    else:
        fig = plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(x)
        plt.title("Input")
        plt.subplot(1, 3, 2)
        plt.imshow(y)
        plt.title("Prediction")
        plt.subplot(1, 3, 3)
        plt.imshow(truth)
        plt.title("Ground Truth")
        plt.suptitle('Patient: {}, DICE: {}'.format(PID, DICE))
        plt.show()


def save_examples(img, seg, gt, PID, fname, example_dir):
        
        # Save binary versions of segmentation masks
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(seg, cmap='gray')
        plt.title("Predicted")
        plt.subplot(1, 2, 2)
        plt.imshow(gt, cmap='gray')
        plt.title("Ground Truth")
        plt.suptitle('PID: {}, Filename: {}'.format(PID, fname))
        fig.savefig(os.path.join(example_dir, f'binarySeg_{PID}_{fname}.png'))
        plt.close(fig)

        # Save contours of segmentations superimposed on input images
        _, gt_countours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        _, pred_countours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        cv2.imwrite(os.path.join(example_dir, 'delete.png'), img * 255)
        anImage = cv2.imread(os.path.join(example_dir, 'delete.png'))
        #anImage = img[x, :, :] * 255

        cv2.drawContours(anImage, gt_countours, -1, (0, 204, 0), 1, cv2.LINE_AA)  #green
        cv2.drawContours(anImage, pred_countours, -1, (0, 0, 204), 1, cv2.LINE_AA)  #red

        #anImage = cv2.flip(anImage, 1)
        
        anImage = cv2.resize(anImage, (anImage.shape[0]*2, anImage.shape[1]*2), interpolation = cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(example_dir, f'{PID}_{fname}.png'), anImage)
        
"""
def bce_fbeta_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + fbeta_loss(y_true, y_pred)
    return loss
"""
