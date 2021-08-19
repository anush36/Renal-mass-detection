import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys
import glob
import time
from tqdm import tqdm
import utils
import pandas as pd
import sklearn.metrics as metrics
import skimage.morphology

class predicter():

    def __init__(self, PARAMS, model, dataset, tmp_dir, og_dir=None, kidney=False, ensemble=None):
        self.kidney = kidney
        self.tmp_dir = tmp_dir
        self.og_dir = og_dir
        self.ensemble = ensemble
        self.Project_Name = PARAMS.Project_Name
        self.IMG_SIZE = PARAMS.IMG_SIZE
        self.BATCH_SIZE = PARAMS.BATCH_SIZE
        self.DOWNSIZE_METHOD = PARAMS.DOWNSIZE_METHOD
        self.model = model
        self.dataset = dataset
        self.precision_list = []
        self.recall_list = []
        self.hausdorff_list = []
        self.hd_pairs = []
        self.abs_vol_dif_list = []
        self.dice_list = []
        self.auc_list = []
        self.centroid_list = []
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0
        
        self.lesion_precision_list = []
        self.lesion_recall_list = []
        self.lesion_hausdorff_list = []
        self.lesion_abs_vol_dif_list = []
        self.lesion_dice_list = []
        self.lesion_auc_list = []
        self.lesion_hd_pairs = []

    def predict_valset(self, save_predictions=False, to_plot=False, threshold=True):
        self.save_predictions = save_predictions
        self.to_plot = to_plot
        self.threshold = threshold
        PIDs = self.dataset.val_slices_in_PID[:,0]
        PID_slices = self.dataset.val_slices_in_PID[:,1]
        test_ds = self.dataset.val_ds

        for k in tqdm(range(PIDs.shape[0])):
            if k == 0:
                current_test = test_ds.take(PID_slices[k])
            else:
                # skip (number of slices in all subsequent PIds)
                # take (number of slices in current PID)
                current_test = test_ds.skip(np.sum(PID_slices[0:k]))
                current_test = current_test.take(PID_slices[k])
            self.run_model(current_test, PIDs[k])


    def predict_testset(self, save_predictions=False, to_plot=False, threshold=True):
        self.save_predictions = save_predictions
        self.to_plot = to_plot
        self.threshold = threshold
        PIDs = self.dataset.test_slices_in_PID[:,0]
        PID_slices = self.dataset.test_slices_in_PID[:,1]
        test_ds = self.dataset.test_ds

        #dice_list = np.zeros(1, dtype=np.float32)
        #precision_list = np.zeros(1, dtype=np.float32)
        #recall_list = np.zeros(1, dtype=np.float32)

        for k in tqdm(range(PIDs.shape[0])):
            if k == 0:
                current_test = test_ds.take(PID_slices[k])
            else:
                # skip (number of slices in all subsequent PIds)
                # take (number of slices in current PID)
                current_test = test_ds.skip(np.sum(PID_slices[0:k]))
                current_test = current_test.take(PID_slices[k])
            self.run_model(current_test, PIDs[k])


    def predict_kfoldvalset(self, i, save_predictions=False, to_plot=False, threshold=False):
        self.save_predictions = save_predictions
        self.to_plot = to_plot
        self.threshold = threshold
        PIDs = self.dataset.val_slices_in_PID[i][:,0]
        PID_slices = self.dataset.val_slices_in_PID[i][:,1]
        val_ds = self.dataset.fold_val_ds[i]

        for k in tqdm(range(PIDs.shape[0])):
            if k == 0:
                current_test = val_ds.take(PID_slices[k])
            else:
                # skip (number of slices in all subsequent PIds)
                # take (number of slices in current PID)
                current_test = val_ds.skip(np.sum(PID_slices[0:k]))
                current_test = current_test.take(PID_slices[k])

            self.run_model(current_test, PIDs[k])


    def run_model(self, current_test, PIDs):
        # current_test includes all the slices for one patient

        if self.TwoChan:
            gt = np.zeros([1, self.IMG_SIZE, self.IMG_SIZE, 1])
            img = np.zeros([1, self.IMG_SIZE, self.IMG_SIZE, 2])

            for element in current_test.as_numpy_iterator():
                elemental_img = element[0][:,:,:]
                elemental_seg = element[1][:,:,:]
                elemental_img = np.expand_dims(elemental_img, axis=0)
                elemental_seg = np.expand_dims(elemental_seg, axis=0)
                gt = np.concatenate([gt, elemental_seg], axis=0)
                img = np.concatenate([img, elemental_img], axis=0)

            # this gets rid of the initial zero entry
            gt = np.squeeze(gt[1:, :, :, :])
            img = img[1:, :, :, :]

        else:
            gt = np.zeros([1, self.IMG_SIZE, self.IMG_SIZE])
            img = np.zeros([1, self.IMG_SIZE, self.IMG_SIZE])

            for element in current_test.as_numpy_iterator():
                elemental = np.array(element)
                elemental_img = elemental[0,:,:,0]
                elemental_seg = elemental[1,:,:,0]
                elemental_img = np.expand_dims(elemental_img, axis=0)
                elemental_seg = np.expand_dims(elemental_seg, axis=0)
                gt = np.concatenate([gt, elemental_seg], axis=0)
                img = np.concatenate([img, elemental_img], axis=0)

            # this gets rid of the initial zero entry
            gt = gt[1:, :, :]
            img = img[1:, :, :]
        
        current_test = current_test.batch(self.BATCH_SIZE)

        predictions = self.model.predict(current_test, verbose=0)
        predictions = np.squeeze(predictions)

        self.auc_list.append(utils.preThres_auc(predictions, gt))

        y = np.where(np.greater_equal(predictions, tf.constant(0.5)), 1, 0)
        
        if self.ensemble != None:

            # Note: ensembles are assumed to use two-channel input methods

            lesion_path = glob.glob(os.path.join(self.ensemble, str(PIDs), 'pred_segs', '*'))
            
            pred_length = len(lesion_path)
            
            for x in range(pred_length):
                    # Ensures that naming convention of predictions matches that of ground truth files for easy loading later
                    if x - 100 < 0:
                        slice_index = '0' + str(x)
                        if x - 10 < 0:
                            slice_index = '0' + slice_index
                    else:
                        slice_index = x
                    fname = "{}.npy".format(slice_index)
                    
                    lesion_path[x] = os.path.join(self.ensemble, str(PIDs), 'pred_segs', fname)
                    
            lesion_pred = np.asarray([np.load(x) for x in lesion_path])
            
            a_TP, a_FP, a_FN, a_cent_dist, lesion_gt, lesion_y = utils.compare_blobs(img[:,:,:,1], predictions, gt, lesion_pred, str(PIDs), self.og_dir, Ensemble=True)
        
            print(f"For case {str(PIDs)}: found {a_TP} TP, {a_FP} FP, {a_FN} FN")
            self.TP += a_TP
            self.FP += a_FP
            self.FN += a_FN
            self.centroid_list.extend(a_cent_dist)
            
            self.lesion_dice_list.append(utils.dice_coeff(lesion_gt, lesion_y).numpy())

            self.lesion_abs_vol_dif_list.append(utils.abs_vol_dif(lesion_y, lesion_gt))
            self.lesion_hausdorff_list.append(utils.max_hausdorff(lesion_y, lesion_gt))
            self.lesion_hd_pairs.append(utils.hd_pairs(lesion_y, lesion_gt))
            
            lesion_y = lesion_y.astype(np.int32)
            lesion_gt = lesion_gt.astype(np.int32)

            self.lesion_precision_list.append(metrics.precision_score(lesion_gt.flatten(), lesion_y.flatten(), zero_division=1))
            self.lesion_recall_list.append(metrics.recall_score(lesion_gt.flatten(), lesion_y.flatten(), zero_division=1))

        if self.kidney == False and self.ensemble == None:

            #NOTE: repeat y terms are to make input size compatible with compare_blobs function
            a_TP, a_FP, a_FN, a_cent_dist, _, _ = utils.compare_blobs(y, y, gt, y, str(PIDs), self.og_dir, Ensemble=False)
        
            self.TP += a_TP
            self.FP += a_FP
            self.FN += a_FN
            self.centroid_list.extend(a_cent_dist)

        if self.kidney:
            print("CLEANING KIDNEY SEG...")
            y = utils.postprocess_kidseg(y)

        if self.to_plot is True:
            example_dir = os.path.join(self.tmp_dir, 'examples')
            Path(example_dir).mkdir(parents=True, exist_ok=True)
            
            if self.ensemble != None:
                for x in range(predictions.shape[0]):
                    utils.save_examples(img[x, :, :, 0], lesion_y[x, :, :], lesion_gt[x, :, :], str(PIDs), x, example_dir)
                    #utils.save_examples(img[x, :, :, 0], lesion_y[x, :, :], lesion_gt[x, :, :], str(PIDs), x, example_dir)
                utils.highlight_blobs(lesion_y, str(PIDs), 'PRED', example_dir)
                utils.highlight_blobs(lesion_gt, str(PIDs), 'GT', example_dir)
                utils.highlight_blobs(img[:,:,:,1], str(PIDs), 'WholeKidney', example_dir)
                
            else:
                for x in range(predictions.shape[0]):
                    utils.save_examples(img[x, :, :, 0], y[x, :, :], gt[x, :, :], str(PIDs), x, example_dir)
                #utils.highlight_blobs(y, str(PIDs), 'PRED', example_dir)
                #utils.highlight_blobs(gt, str(PIDs), 'GT', example_dir)
                
            self.to_plot = False

        if self.save_predictions is True:

            prediction_img_dir = os.path.join(self.tmp_dir, 'results/', str(PIDs), 'img')
            prediction_seg_dir = os.path.join(self.tmp_dir, 'results/', str(PIDs), 'pred_segs')
            Path(prediction_img_dir).mkdir(parents=True, exist_ok=True)
            Path(prediction_seg_dir).mkdir(parents=True, exist_ok=True)

            if self.ensemble != None:
                for x in range(predictions.shape[0]):
                    # Ensures that naming convention of predictions matches that of ground truth files for easy loading later
                    if x - 100 < 0:
                        slice_index = '0' + str(x)
                        if x - 10 < 0:
                            slice_index = '0' + slice_index
                    else:
                        slice_index = x
                    fname = "{}.npy".format(slice_index)
                    
                    np.save(os.path.join(prediction_seg_dir, fname), waste_pred[x, :, :])
                    np.save(os.path.join(prediction_img_dir, fname), img[x, :, :, :])
            else:
                for x in range(predictions.shape[0]):
                    # Ensures that naming convention of predictions matches that of ground truth files for easy loading later
                    if x - 100 < 0:
                        slice_index = '0' + str(x)
                        if x - 10 < 0:
                            slice_index = '0' + slice_index
                    else:
                        slice_index = x
                    fname = "{}.npy".format(slice_index)

                    np.save(os.path.join(prediction_seg_dir, fname), predictions[x, :, :])
                    np.save(os.path.join(prediction_img_dir, fname), img[x, :, :, :])

        if self.threshold is True:
            self.dice_list.append(utils.dice_coeff(gt, y).numpy())
        else:
            self.dice_list.append(utils.dice_coeff(gt, predictions).numpy())

        self.abs_vol_dif_list.append(utils.abs_vol_dif(y, gt))
        self.hausdorff_list.append(utils.max_hausdorff(y, gt))

        y = y.astype(np.int32)
        gt = gt.astype(np.int32)

        self.precision_list.append(metrics.precision_score(gt.flatten(), y.flatten(), zero_division=1))
        self.recall_list.append(metrics.recall_score(gt.flatten(), y.flatten(), zero_division=1))
        
        return
