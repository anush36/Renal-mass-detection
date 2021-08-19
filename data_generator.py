import tensorflow as tf
#import elasticdeform as etf
import numpy as np
import os
import pathlib
import sys
import utils
import glob
# NOTES



class theDataset():

    seg_const = os.path.join('seg')

    def __init__(self, origin, PARAMS, TwoChan, balancing=False):
        self.origin_tf = tf.constant(origin)
        self.origin = origin
        self.IMG_SIZE = PARAMS.IMG_SIZE
        self.BATCH_SIZE = PARAMS.BATCH_SIZE
        self.ELASTIC_CHANCE = PARAMS.ELASTIC_CHANCE
        self.ELASTIC_SIGMA = PARAMS.ELASTIC_SIGMA
        self.DOWNSIZE_METHOD = PARAMS.DOWNSIZE_METHOD
        self.balancing = balancing
        self.TwoChan = TwoChan


    def file_io(self, filenames):

        aImage = np.load(filenames)
        parts = tf.strings.split(filenames, os.path.sep)
        if self.balancing == True:
            label_path = tf.strings.join([self.origin_tf, os.path.sep, parts[-5], os.path.sep, parts[-4], os.path.sep, parts[-3], os.path.sep, self.seg_const, os.path.sep, parts[-1]])
        else:
            label_path = tf.strings.join([self.origin_tf, os.path.sep, parts[-4], os.path.sep, parts[-3], os.path.sep, self.seg_const, os.path.sep, parts[-1]])
        label_path = label_path.numpy()
        aMask = np.load(label_path)


        # DO THE ELASTIC DEFORMATION HERE
        if np.random.uniform() < self.ELASTIC_CHANCE:

            if self.TwoChan:
                [defImage, defKidney, defMask] = utils.deform_grid(X=aImage[:,:,0], Z=aImage[:,:,1], Y=aMask)
            else:
                [defImage, defMask] = utils.deform_grid(X=aImage, Y=aMask)
        else:

            if self.TwoChan:
                defImage = aImage[:,:,0]
                defKidney = aImage[:,:,1]
                defMask = aMask
            else:
                defImage = aImage
                defMask = aMask

        # Note: label is already normalized (binary)
        defImage = defImage - np.amin(defImage)
        if np.amax(defImage) != 0:
            defImage = defImage / np.amax(defImage)

        if self.TwoChan:
            defImage = np.expand_dims(defImage, axis = -1)
            defKidney = np.expand_dims(defKidney, axis = -1)
            defImage = np.concatenate((defImage, defKidney), axis = -1) 

        image = tf.convert_to_tensor(defImage, dtype=tf.float32)
        mask = tf.convert_to_tensor(defMask, dtype=tf.float32)

        return image, mask

    def file_io_test(self, filenames):

        aImage = np.load(filenames)

        parts = tf.strings.split(filenames, os.path.sep)
        if self.balancing == True:
            label_path = tf.strings.join([self.origin_tf, os.path.sep, parts[-5], os.path.sep, parts[-4], os.path.sep, parts[-3], os.path.sep, self.seg_const, os.path.sep, parts[-1]])
        else:
            label_path = tf.strings.join([self.origin_tf, os.path.sep, parts[-4], os.path.sep, parts[-3], os.path.sep, self.seg_const, os.path.sep, parts[-1]])
        label_path = label_path.numpy()

        aMask = np.load(label_path)

        if self.TwoChan:
            aKidney = aImage[:,:,1]
            aImage = aImage[:,:,0]

        #Note: label is already normalized (binary)
        aImage = aImage - np.amin(aImage)
        if np.amax(aImage) != 0:
            aImage = aImage / np.amax(aImage)

        if self.TwoChan:
            aImage = np.expand_dims(aImage, axis = -1)
            aKidney = np.expand_dims(aKidney, axis = -1)
            aImage = np.concatenate((aImage, aKidney), axis = -1) 

        image = tf.convert_to_tensor(aImage, dtype = tf.float32)
        mask = tf.convert_to_tensor(aMask, dtype = tf.float32)

        return image, mask

    def gen_series(self,filenames):
        # This contains mappings common to the entire series of training data (train set + val set). It opens the numpy array file and converts it to a tensor for further processing.

        image, mask = tf.numpy_function(self.file_io, [filenames], [tf.float32, tf.float32])

        #add channel dimension to the image. (This is just binary classificaiton so #channels = 1)
        mask = tf.expand_dims(mask, -1)
        if self.TwoChan == False:
            image = tf.expand_dims(image, -1)

        #first make the image box shaped
        asize = tf.math.reduce_max(tf.shape(mask))
        mask = tf.image.resize_with_crop_or_pad(mask, asize, asize)
        image = tf.image.resize_with_crop_or_pad(image, asize, asize)

        #now downsample to the image size.
        mask = tf.image.resize(mask, [self.IMG_SIZE, self.IMG_SIZE], method='nearest')
        image = tf.image.resize(image, [self.IMG_SIZE, self.IMG_SIZE], method=self.DOWNSIZE_METHOD)

        return image, mask

    def gen_series_test(self,filenames):
        # This contains mappings common to the test sets only. It opens the numpy array file and converts it to a tensor for further processing.

        image, mask = tf.numpy_function(self.file_io_test, [filenames], [tf.float32, tf.float32])

        #add channel dimension to the image. (This is just binary classificaiton so #channels = 1)
        mask = tf.expand_dims(mask, -1)
        if self.TwoChan == False:
            image = tf.expand_dims(image, -1)

        #standardize image size
        #first make the image box shaped
        asize = tf.math.reduce_max(tf.shape(mask))
        mask = tf.image.resize_with_crop_or_pad(mask, asize, asize)
        image = tf.image.resize_with_crop_or_pad(image, asize, asize)

        #now downsample to the image size.
        mask = tf.image.resize(mask, [self.IMG_SIZE, self.IMG_SIZE], method = 'nearest')
        image = tf.image.resize(image, [self.IMG_SIZE, self.IMG_SIZE], method = self.DOWNSIZE_METHOD)

        return image, mask

    def PID_isolator(self, list_ds):
        PIDs = []

        for path in list_ds:
            path = path.numpy().decode("utf-8")
            locs = path.split('/')
            
            if locs[-1] not in PIDs:
                PIDs.append(locs[-1])
            
        return tf.constant(PIDs)

    def PID_to_file_list(self, origin, PID_List):
        """
        ###### CONVERT PID_List list of PIDs INTO a_list_ds DATASET WITH PATHS TO ALL IMAGES OF PIDs ############

        slices_in_PID - numpy array containing [PID, #ofslices] arranged in rows.
        """
        parts = origin.split(os.path.sep)

        PID_size = int(tf.shape(PID_List).numpy())
        counter = 0

        if self.balancing == True and parts[-1] == 'train':
            pos_list_ds_np = np.empty(PID_size, dtype=object)
            neg_list_ds_np = np.empty(PID_size, dtype=object)

            for patient in PID_List:
                pos_data_dir = os.path.join(origin, patient.numpy().decode(), 'positive', 'images/')
                neg_data_dir = os.path.join(origin, patient.numpy().decode(), 'negative', 'images/')
                pos_data_dir = pathlib.Path(pos_data_dir)
                neg_data_dir = pathlib.Path(neg_data_dir)

                pos_list_ds_np[counter] = tf.data.Dataset.list_files(str(pos_data_dir/'*.npy'), shuffle=False)
                neg_list_ds_np[counter] = tf.data.Dataset.list_files(str(neg_data_dir/'*.npy'), shuffle=False)

                counter += 1

            pos_list_ds = pos_list_ds_np[0]
            neg_list_ds = neg_list_ds_np[0]

            for entry in range(1, PID_size):
                pos_list_ds = pos_list_ds.concatenate(pos_list_ds_np[entry])
                neg_list_ds = neg_list_ds.concatenate(neg_list_ds_np[entry])

            negs = tf.data.experimental.cardinality(neg_list_ds).numpy()
            poses = tf.data.experimental.cardinality(pos_list_ds).numpy()

            neg_list_ds = neg_list_ds.repeat()
            pos_list_ds = pos_list_ds.repeat()

            balanced_list_ds = tf.data.experimental.sample_from_datasets([neg_list_ds, pos_list_ds], [0.5, 0.5])

            return balanced_list_ds, negs

        else:
            a_list_ds_np = np.empty(PID_size, dtype=object)

            slices_in_PID = np.zeros((PID_size, 2), dtype=object)
            

            for patient in PID_List:

                if self.balancing == True:
                    data_dir = os.path.join(origin, patient.numpy().decode(),'neutral', 'images/')
                else:
                    data_dir = os.path.join(origin, patient.numpy().decode(),'images/')

                data_dir = pathlib.Path(data_dir)

                a_list_ds_np[counter] = tf.data.Dataset.list_files(str(data_dir/'*.npy'), shuffle = False)
                """
                if counter == 2:
                    print("----ORDER OF FILES LOADED-----"
                    for stuff in a_list_ds_np[counter].take(-1):
                        print(stuff)
                """
                slices_in_PID[counter, 0] = patient.numpy().decode()
                slices_in_PID[counter, 1] = tf.data.experimental.cardinality(a_list_ds_np[counter]).numpy()

                counter += 1

            #this line initializes the dataset for the loop
            a_list_ds = a_list_ds_np[0]

            for entry in range(1, PID_size):
                a_list_ds = a_list_ds.concatenate(a_list_ds_np[entry])

            return a_list_ds, slices_in_PID

    def kfold_PID_to_file_list(self, origin, PID_List, is_test=False):
        """
        ###### CONVERT PID_List list of PIDs INTO a_list_ds DATASET WITH PATHS TO ALL IMAGES OF PIDs ############

        slices_in_PID - numpy array containing [PID, #ofslices] arranged in rows.
        """
        PID_size = int(tf.shape(PID_List).numpy())

        a_list_ds_np = np.empty(PID_size, dtype=object)

        slices_in_PID = np.zeros((PID_size, 2), dtype=object)
        counter = 0

        if self.balancing == True and is_test == False:
            pos_list_ds_np = np.empty(PID_size, dtype=object)
            neg_list_ds_np = np.empty(PID_size, dtype=object)
            
            for patient in PID_List:
                pos_data_dir = os.path.join(origin, 'train', patient.numpy().decode(), 'positive', 'images/')
                neg_data_dir = os.path.join(origin, 'train', patient.numpy().decode(), 'negative', 'images/')
                pos_data_dir = pathlib.Path(pos_data_dir)
                neg_data_dir = pathlib.Path(neg_data_dir)
                
                exist_check = glob.glob(os.path.join(neg_data_dir, '*.npy'))
                
                if len(exist_check) < 2:
                    pos_data_dir = os.path.join(origin, 'val', patient.numpy().decode(), 'positive', 'images/')
                    neg_data_dir = os.path.join(origin, 'val', patient.numpy().decode(), 'negative', 'images/')
                    pos_data_dir = pathlib.Path(pos_data_dir)
                    neg_data_dir = pathlib.Path(neg_data_dir)
                    
                exist_check = glob.glob(os.path.join(neg_data_dir, '*.npy'))
                
                if len(exist_check) < 2:
                    pos_data_dir = os.path.join(origin, 'test', patient.numpy().decode(), 'positive', 'images/')
                    neg_data_dir = os.path.join(origin, 'test', patient.numpy().decode(), 'negative', 'images/')
                    pos_data_dir = pathlib.Path(pos_data_dir)
                    neg_data_dir = pathlib.Path(neg_data_dir)
                
                pos_list_ds_np[counter] = tf.data.Dataset.list_files(str(pos_data_dir/'*.npy'), shuffle=False)
                neg_list_ds_np[counter] = tf.data.Dataset.list_files(str(neg_data_dir/'*.npy'), shuffle=False)
                
                counter += 1
            
            pos_list_ds = pos_list_ds_np[0]
            neg_list_ds = neg_list_ds_np[0]

            for entry in range(1, PID_size):
                pos_list_ds = pos_list_ds.concatenate(pos_list_ds_np[entry])
                neg_list_ds = neg_list_ds.concatenate(neg_list_ds_np[entry])

            negs = tf.data.experimental.cardinality(neg_list_ds).numpy()
            poses = tf.data.experimental.cardinality(pos_list_ds).numpy()

            neg_list_ds = neg_list_ds.repeat()
            pos_list_ds = pos_list_ds.repeat()

            balanced_list_ds = tf.data.experimental.sample_from_datasets([neg_list_ds, pos_list_ds], [0.5, 0.5])

            return balanced_list_ds, negs
            
        elif self.balancing == True and is_test == True:
                
            for patient in PID_List:

                data_dir = os.path.join(origin, 'train', patient.numpy().decode(),'neutral', 'images/')
                data_dir = pathlib.Path(data_dir)

                exist_check = glob.glob(os.path.join(data_dir, '*.npy'))
            
                if len(exist_check) < 2:
                    data_dir = os.path.join(origin, 'val', patient.numpy().decode(),'neutral', 'images/')
                    data_dir = pathlib.Path(data_dir)
                
                exist_check = glob.glob(os.path.join(data_dir, '*.npy'))
            
                if len(exist_check) < 2:
                    data_dir = os.path.join(origin, 'test', patient.numpy().decode(),'neutral', 'images/')
                    data_dir = pathlib.Path(data_dir)
                    
                a_list_ds_np[counter] = tf.data.Dataset.list_files(str(data_dir/'*.npy'), shuffle = False)
            
                slices_in_PID[counter, 0] = patient.numpy().decode()
                slices_in_PID[counter, 1] = tf.data.experimental.cardinality(a_list_ds_np[counter]).numpy()

                counter += 1

            #this line initializes the dataset for the loop
            a_list_ds = a_list_ds_np[0]

            for entry in range(1, PID_size):
                a_list_ds = a_list_ds.concatenate(a_list_ds_np[entry])

            return a_list_ds, slices_in_PID

        else:
                
            for patient in PID_List:

                data_dir = os.path.join(origin, 'train', patient.numpy().decode(),'images/')
                data_dir = pathlib.Path(data_dir)

                exist_check = glob.glob(os.path.join(data_dir, '*.npy'))
            
                if len(exist_check) < 2:
                    data_dir = os.path.join(origin, 'val', patient.numpy().decode(),'images/')
                    data_dir = pathlib.Path(data_dir)
                    
                exist_check = glob.glob(os.path.join(data_dir, '*.npy'))
            
                if len(exist_check) < 2:
                    data_dir = os.path.join(origin, 'test', patient.numpy().decode(),'images/')
                    data_dir = pathlib.Path(data_dir)

                a_list_ds_np[counter] = tf.data.Dataset.list_files(str(data_dir/'*.npy'), shuffle = False)
            
                slices_in_PID[counter, 0] = patient.numpy().decode()
                slices_in_PID[counter, 1] = tf.data.experimental.cardinality(a_list_ds_np[counter]).numpy()

                counter += 1

            #this line initializes the dataset for the loop
            a_list_ds = a_list_ds_np[0]

            for entry in range(1, PID_size):
                a_list_ds = a_list_ds.concatenate(a_list_ds_np[entry])

            return a_list_ds, slices_in_PID

    def conventional_split(self):
    
        train_list_ds = tf.data.Dataset.list_files(str(os.path.join(self.origin, 'train', '*')), shuffle = False)
        val_list_ds = tf.data.Dataset.list_files(str(os.path.join(self.origin, 'val', '*')), shuffle = False)
        test_list_ds = tf.data.Dataset.list_files(str(os.path.join(self.origin, 'test', '*')), shuffle = False)
        
        self.train_set = self.PID_isolator(train_list_ds)
        self.val_set = self.PID_isolator(val_list_ds)
        self.test_set = self.PID_isolator(test_list_ds)

        print(f"------------------------The PIDs in the train_set are {self.train_set}--------------------")
        print(f"------------------------The PIDs in the test_set are {self.test_set}--------------------")

        test_list_ds, self.test_slices_in_PID = self.PID_to_file_list(os.path.join(self.origin, 'test'), self.test_set)
        train_list_ds, self.train_size = self.PID_to_file_list(os.path.join(self.origin, 'train'), self.train_set)
        val_list_ds, self.val_slices_in_PID = self.PID_to_file_list(os.path.join(self.origin, 'val'), self.val_set)

        self.test_ds = test_list_ds.map(self.gen_series_test)
        self.train_ds = train_list_ds.map(self.gen_series)
        self.val_ds = val_list_ds.map(self.gen_series_test)

    def k_fold_split(self, k):
        
        train_list_ds = tf.data.Dataset.list_files(str(os.path.join(self.origin, 'train', '*')), shuffle = False)
        val_list_ds = tf.data.Dataset.list_files(str(os.path.join(self.origin, 'val', '*')), shuffle = False)
        test_list_ds = tf.data.Dataset.list_files(str(os.path.join(self.origin, 'test', '*')), shuffle = False)

        self.train_set = self.PID_isolator(train_list_ds)
        self.val_set = self.PID_isolator(val_list_ds)
        self.train_set = tf.concat([self.train_set, self.val_set], 0)

        self.test_set = self.PID_isolator(test_list_ds)
        test_list_ds, self.test_slices_in_PID = self.PID_to_file_list(os.path.join(self.origin, 'test'), self.test_set)

        self.fold_train_ds = np.empty(k, dtype=object)
        self.fold_val_ds = np.empty(k, dtype=object)
        self.val_slices_in_PID = np.empty(k, dtype=object)

        self.train_set = tf.random.shuffle(self.train_set, seed = 956)
        PIDs_shuffled = self.train_set.numpy()

        PID_size = len(PIDs_shuffled)
        PID_fold_size = int(np.round(PID_size / k))

        for i in range(k):
            if i == (k - 1):
                self.fold_val_ds[i] = PIDs_shuffled[PID_fold_size*i:]
            else:
                self.fold_val_ds[i] = PIDs_shuffled[PID_fold_size*i : PID_fold_size*(i+1)]

            self.fold_train_ds[i] = np.setdiff1d(PIDs_shuffled, self.fold_val_ds[i])
            self.fold_train_ds[i] = tf.constant(self.fold_train_ds[i])
            self.fold_val_ds[i] = tf.constant(self.fold_val_ds[i])

            print(f"------------------------The PIDs in the train_set are {self.fold_train_ds[i]}--------------------")
            print(f"------------------------The PIDs in the val_set are {self.fold_val_ds[i]}--------------------")
        
            train_list_ds, _ = self.kfold_PID_to_file_list(self.origin, self.fold_train_ds[i])
            val_list_ds, self.val_slices_in_PID[i] = self.kfold_PID_to_file_list(self.origin, self.fold_val_ds[i], is_test=True)

            self.fold_val_ds[i] = val_list_ds.map(self.gen_series_test)
            self.fold_train_ds[i] = train_list_ds.map(self.gen_series)

    def full_dataset_k_fold_split(self, k):
        
        train_list_ds = tf.data.Dataset.list_files(str(os.path.join(self.origin, 'train', '*')), shuffle = False)
        val_list_ds = tf.data.Dataset.list_files(str(os.path.join(self.origin, 'val', '*')), shuffle = False)
        test_list_ds = tf.data.Dataset.list_files(str(os.path.join(self.origin, 'test', '*')), shuffle = False)

        self.train_set = self.PID_isolator(train_list_ds)
        self.val_set = self.PID_isolator(val_list_ds)
        self.test_set = self.PID_isolator(test_list_ds)
        self.train_set = tf.concat([self.train_set, self.val_set, self.test_set], 0)

        
        #test_list_ds, self.test_slices_in_PID = self.PID_to_file_list(os.path.join(self.origin, 'test'), self.test_set)

        self.fold_train_ds = np.empty(k, dtype=object)
        self.fold_val_ds = np.empty(k, dtype=object)
        self.val_slices_in_PID = np.empty(k, dtype=object)

        self.train_set = tf.random.shuffle(self.train_set, seed = 956)
        PIDs_shuffled = self.train_set.numpy()

        PID_size = len(PIDs_shuffled)
        PID_fold_size = int(np.round(PID_size / k))

        for i in range(k):
            if i == (k - 1):
                self.fold_val_ds[i] = PIDs_shuffled[PID_fold_size*i:]
            else:
                self.fold_val_ds[i] = PIDs_shuffled[PID_fold_size*i : PID_fold_size*(i+1)]

            self.fold_train_ds[i] = np.setdiff1d(PIDs_shuffled, self.fold_val_ds[i])
            self.fold_train_ds[i] = tf.constant(self.fold_train_ds[i])
            self.fold_val_ds[i] = tf.constant(self.fold_val_ds[i])

            print(f"------------------------The PIDs in the train_set are {self.fold_train_ds[i]}--------------------")
            print(f"------------------------The PIDs in the val_set are {self.fold_val_ds[i]}--------------------")
        
            train_list_ds, self.train_size = self.kfold_PID_to_file_list(self.origin, self.fold_train_ds[i])
            val_list_ds, self.val_slices_in_PID[i] = self.kfold_PID_to_file_list(self.origin, self.fold_val_ds[i], is_test=True)

            self.fold_val_ds[i] = val_list_ds.map(self.gen_series_test)
            self.fold_train_ds[i] = train_list_ds.map(self.gen_series)
