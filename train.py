import sys
import numpy as np
import pandas as pd
from pathlib import Path
import os
import tensorflow as tf
from attrdict import AttrDict
import json
import glob
import subprocess

import data_generator
from models import unet_vanilla
import utils

#Set 5-fold cross validation interation
i = int(sys.argv[1])

#Path to pre-processed data
tmp_dir = subprocess.getoutput('echo $SLURM_TMPDIR')
origin = os.path.join(tmp_dir, 'data', 'ngmri_wholeKidney_adahist_voxnorm')

#HYPERPARAMETER LIST
PARAMS = {'Project_Name': 'bcedice_medphys_kidseg_wholeNGMRI' + f"_{i}",
          'IMG_SIZE': 256,
          'BATCH_SIZE': 36,
          'EPOCHS': 75,
          'ELASTIC_CHANCE': 0.25,
          'ELASTIC_SIGMA': 15,
          'DOWNSIZE_METHOD': "bilinear",   # Needs to match tensorflow formatting. See: https://www.tensorflow.org/api_docs/python/tf/image/ResizeMethod
          'DROPOUT': 0.2,
          'LEARNING_RATE': 0.001,
          'FBETA': 1.0, 
          'KFOLD': i
          }

PARAMS = AttrDict(PARAMS)
utils.PARAMS = PARAMS

weight_path = "weights"
save_model_path = os.path.join(weight_path, PARAMS.Project_Name + '.hdf5')

#Initialize data generator with hyperparameters and data split settings
mri_dataset = data_generator.theDataset(origin = origin, PARAMS = PARAMS, TwoChan=True)
mri_dataset.full_dataset_k_fold_split(5)

#Isolate training set from data generator and prepare for model
train_ds = mri_dataset.fold_train_ds[PARAMS.KFOLD]
train_size = tf.data.experimental.cardinality(train_ds).numpy()
train_ds = train_ds.shuffle(400).batch(PARAMS.BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

#Initialize model architecture, choose loss configuration
model = unet_vanilla.unet_model(PARAMS.IMG_SIZE, PARAMS.DROPOUT)
opt = tf.keras.optimizers.Adam(learning_rate = PARAMS.LEARNING_RATE)
model.compile(optimizer=opt, loss=utils.bce_dice_loss, metrics=[utils.dice_loss])

cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='loss', save_best_only=True, verbose=1)
es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, min_delta = 0.00001, restore_best_weights=True)

#Save hyperparameters for records
json = json.dumps(PARAMS)
f = open(os.path.join(weight_path, PARAMS.Project_Name + ".json"),"w")
f.write(json)
f.close()

print("got to the model training step!")

#Train ML model
history = model.fit(train_ds, 
                   steps_per_epoch=int(np.ceil(train_size / float(PARAMS.BATCH_SIZE))),
                   epochs=PARAMS.EPOCHS,
                   verbose=2,
                   callbacks=[cp, es])

#Save training history
hist_df = pd.DataFrame(history.history)
hist_csv_file = os.path.join(weight_path, 'history' + f"_{PARAMS.KFOLD}" + '.csv')
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)