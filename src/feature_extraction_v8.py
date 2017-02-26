# -*- coding: utf-8 -*-
"""
This class generate statiscal feature of 1st derivative
"""

import warnings

import pandas as pd
import scipy.stats.stats as st
from spectrum import *

from feature_extraction import iqr, mad, energy
from visualise_data import Sequence
warnings.filterwarnings('ignore')


feature_functions = [np.mean, np.std, np.min, np.median, np.max, np.var, st.skew, st.kurtosis,
                     st.sem,st.moment,iqr, mad, energy,np.linalg.norm]
feature_names = ['mean', 'std', 'min', 'median', 'max', 'var', 'skew', 'kur',
                 'sem', 'moment','iqr','mad','energy','mag']
                 
feature_names = map(lambda x: "diff_%s" % x, feature_names)

# We will keep the number of extracted feature functions as a parameter 
num_ff = len(feature_functions)

# We will want to keep track of the feature names for later, so we will collect these in the following list: 
column_names = []

# These are the modalities that are available in the dataset, and the .iterate() function returns the data in this order
modality_names = ['acceleration', 'rssi', 'pir', 'video_living_room', 'video_kitchen', 'video_hallway']

"""
Iterate over all training directories
"""
for train_test in ('train','test'):
    if train_test is 'train':
        print ('Extracting features from training data.\n')
    else:
        print ('\n\n\nExtracting features from testing data.\n')

    for fi, file_id in enumerate(os.listdir('../input/public_data/{}/'.format(train_test))):
        stub_name = str(file_id).zfill(5)

        if train_test is 'train' or np.mod(fi, 50) == 0:
            print ("Starting feature extraction for {}/{}".format(train_test, stub_name))

        # Use the sequence loader to load the data from the directory.
        data = Sequence('../input/public_data', '../input/public_data/{}/{}'.format(train_test, stub_name))
        data.load()

        if len(column_names) == 0:
            for lu, modalities in data.iterate():
                for i, (modality, modality_name) in enumerate(zip(modalities, modality_names)):
                    
                    for column_name, column_data in modality.transpose().iterrows():
                        for feature_name in feature_names:
                            column_names.append('{0}_{1}_{2}'.format(modality_name, column_name, feature_name))
                    

                break

        rows = []

        for ri, (lu, modalities) in enumerate(data.iterate()):
            row = []
            for i, modality in enumerate(modalities):
                # 1st derivative
                modality = modality.diff()
                
                for name, column_data in modality.transpose().iterrows():
                    if len(column_data) > 3:
                        row.extend(map(lambda ff: ff(column_data), feature_functions))
                    else:
                        row.extend([np.nan] * num_ff)
                
            # Do a quick sanity check to ensure that the feature names and number of extracted features match
            assert len(row) == len(column_names)
            # Append the row to the full set of features
            rows.append(row)

            # Report progress
            if train_test is 'train':
                if np.mod(ri + 1, 50) == 0:
                    print ("{:5}".format(str(ri + 1))),

            if np.mod(ri + 1, 500) == 0:
                print
        df = pd.DataFrame(rows)
        df.columns = column_names

        df.to_csv('../input/public_data/{}/{}/columns_v8.csv'.format(train_test, stub_name),
                  index=False)
        if train_test is 'train': print
        print ("Finished feature extraction for {}/{}\n".format(train_test, stub_name))
