# Renal-mass-detection
Fully automated detection of renal masses in MRI using deep learning. Algorithm written in Python using Keras functional-API with a TensorFlow back-end. Work submitted to the Medical Physics peer-reviewed journal, currently under review. Abstract is provided below:

**Title: Deep Learning-Based Ensemble Method for Fully Automated Detection of Renal Masses on Magnetic Resonance Images**\
Authors: Anush Agarwal, Dr. Nicola Schieda, Dr. Mohamed Elfaal, Dr. Eranga Ukwatta

**Purpose:** Accurate detection of renal masses is a fundamental first step for attempted automated classification of benign and malignant or indolent and aggressive renal tumors in the future. MRI may outperform CT for differentiation of renal mass subtype due to improved tissue characterization, but is less well studied compared to CT. The objective of this study is to autonomously detect renal masses on contrast-enhanced magnetic resonance images (CE-MRI).\
**Method:** In this paper, we describe a novel, fully automated methodology for accurate detection and localization of renal masses on CE-MRI. We first determine the boundaries of the kidneys using a U-Net convolutional neural network. The kidney boundary is used as a localizer to identify a region of interest to search for renal masses. We then utilized a mixture-of-experts (MoE) ensemble model based on the U-Net architecture to identify renal masses. Our dataset is comprised of CE-MRI scans of 118 patients with benign and malignant solid kidney tumors including: renal cell carcinoma (clear cell, papillary, chromophobe), oncocytomas, and fat poor renal angiomyolipoma (fpAML). We trained and evaluated the proposed model on the entire CE-MRI dataset using 5-fold cross validation.\
**Results:** The developed algorithm reported a Dice similarity coefficient (DSC) of 91.20 ± 5.41% (mean ± standard deviation) for segmentation accuracy of kidney boundary delineation from 118 volumes consisting of 25,025 slices. Our proposed ensemble model for renal mass detection yielded a recall and precision of 86.2% and 83.3% on the entire CE-MRI dataset, respectively.\
**Conclusion:** We describe a deep learning based method for fully automated renal mass detection using CE-MR images which has not been studied previously. The results demonstrate the usefulness of our suggested technique for this application, which is clinically important as renal mass localization is a pre-step for fully-automated diagnosis of renal mass subtype which may be more accurate on MRI compared to CT.

## Research Presentation Video

[![Renal Mass Detection](https://img.youtube.com/vi/yQcqOi6vQ84/0.jpg)](https://www.youtube.com/watch?v=yQcqOi6vQ84 "Renal Mass Detection")

## File Organization/Features

* submit_kfoldtrain.sh
	* example batch job submission to Compute Canada cluster using SLURM
	* allocates resources and runs 5-fold cross validation in parallel using remote resources
* preprocessing
	* convert DICOM/NIfTI medical images to numpy arrays
	* apply voxel size normalization, intensity normalization, histogram equalization
	* standardize image size for machine learning model
	* Create_KidneySeg.py
		* dataset for binary kidney segmentation task
	* Create_LesionSeg.py
		* dataset for binary renal mass segmentation task
		* uses output from kidney segmentation model to localize region of interest
* data_generator.py
	* split data into train/val/test or k-fold cross validation
	* generate images at run-time for machine learning model
	* apply data augmentation (elastic deformation) stochastically at run-time
	* over-sample unbalanced image classes to address class imbalance
* pred_generator.py
	* generate predictions using trained model
	* post-process predictions
	* evaluate model based on pre-defined metrics
	* save raw prediction files
	* visualize predicted images
* utils.py
	* segmentation and evaluation functions:
		* Dice Similarity Coefficient
		* Hausdorff Distance
		* Relative Absolute Volume Difference
		* Precision/Recall
	* loss functions
		* Dice loss
		* Binary cross-entropy loss
		* Asymmetric Similarity loss
		* visualization functions
	* visualization functions
		* 2D overlay of predicted and ground truth boundaries
		* 3D NRRD files for visualization with Slicer
* models
	* unet_vanilla.py
		* U-Net implementation
	* unet_3d.py
		* 3D U-Net implementation
	* unet_attn2d.py
		* Attention U-Net implementation
* train.py
	* initialize machine learning model training
	* select model architecture
	* configure hyperparameters
	* configure training settings
	* save model weights, hyperparameters, history
* predict.py
	* initialize prediction
	* load weights and test set
	* select options for post-processing, visualization
