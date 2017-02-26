# -*- coding: utf-8 -*-
"""
This class generate accel features
"""
import warnings

import pandas as pd
import scipy.stats.stats as st
from scipy import spatial
from spectrum import *

from feature_extraction import iqr, mad, energy, zero_crossing, peak, entropy
from visualise_data import Sequence
warnings.filterwarnings('ignore')

feature_functions = [np.mean, np.std, np.min, np.median, np.max, np.var, st.skew, st.kurtosis,
                     st.sem,st.moment,iqr, mad, energy,np.linalg.norm]
feature_names = ['mean', 'std', 'min', 'median', 'max', 'var', 'skew', 'kur',
                 'sem', 'moment','iqr','mad','energy','mag']
# We will keep the number of extracted feature functions as a parameter 
num_ff = len(feature_functions)

# We will want to keep track of the feature names for later, so we will collect these in the following list: 
column_names = []

# These are the modalities that are available in the dataset, and the .iterate() function returns the data in this order
modality_names = ['acceleration', 'rssi', 'pir', 'video_living_room', 'video_kitchen', 'video_hallway']

#acceleration feature
feature_accel = ['sma','corrxy','corrxz','corryz','cosinxy','cosinxz','cosinyz'
                            ,'arCoeffX1','arCoeffX2','arCoeffX3','arCoeffX4'
                            ,'arCoeffY1','arCoeffY2','arCoeffY3','arCoeffY4'
                            ,'arCoeffZ1','arCoeffZ2','arCoeffZ3','arCoeffZ4'
                            ,'zero_crossingX','zero_crossingY','zero_crossingZ'
                            ,'peakX','peakY','peakZ'
                            ,'entropyX','entropyY','entropyZ'

                 ]
num_ff_accel = len(feature_accel)

def extract_accel(accel):
    if(accel.shape[0]<5): #insufficient
        return [np.nan] * num_ff_accel
    ff_accel = []
    #Signal-Magnitude Area
    accel['sma'] =  np.sqrt(accel['x']**2 + accel['y']**2 + accel['z']**2)
    sma = accel['sma'].sum()/accel.shape[0]
    ff_accel.append(sma)
    #correlation
    corrxy = st.pearsonr(accel['x'], accel['y'])[0]
    ff_accel.append(corrxy)
    corrxz = st.pearsonr(accel['x'], accel['z'])[0]
    ff_accel.append(corrxz)
    corryz = st.pearsonr(accel['y'], accel['z'])[0]
    ff_accel.append(corryz)
    #angle
    cosinxy = spatial.distance.cosine(accel['x'], accel['y'])
    ff_accel.append(cosinxy)
    cosinxz = spatial.distance.cosine(accel['x'], accel['z'])
    ff_accel.append(cosinxz)
    cosinyz = spatial.distance.cosine(accel['y'], accel['z'])
    ff_accel.append(cosinyz)

    #Autorregresion coefficients with Burg order equal to 4
    arCoeffX, k, e = arburg(accel['x'], 4)
    ff_accel.extend(np.array(arCoeffX, dtype=float))
    arCoeffY, k, e = arburg(accel['y'], 4)
    ff_accel.extend(np.array(arCoeffY, dtype=float))
    arCoeffZ, k, e = arburg(accel['z'], 4)
    ff_accel.extend(np.array(arCoeffZ, dtype=float))

    #zero crossing
    zero_crossingX = zero_crossing(accel['x'])
    ff_accel.append(zero_crossingX)
    zero_crossingY = zero_crossing(accel['y'])
    ff_accel.append(zero_crossingY)
    zero_crossingZ = zero_crossing(accel['z'])
    ff_accel.append(zero_crossingZ)
    #peak
    peakX = peak(accel['x'])
    ff_accel.append(peakX)
    peakY = peak(accel['y'])
    ff_accel.append(peakY)
    peakZ = peak(accel['z'])
    ff_accel.append(peakZ)

    #entropy
    entropyX = entropy(accel['x'])
    ff_accel.append(entropyX)
    entropyY = entropy(accel['y'])
    ff_accel.append(entropyY)
    entropyZ = entropy(accel['z'])
    ff_accel.append(entropyZ)

    return ff_accel

accel_feature_functions = [extract_accel]

for train_test in ('train', 'test',):
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

        """
        Populate the column_name list here. This needs to only be done on the first iteration
        because the column names will be the same between datasets.
        """
        if len(column_names) == 0:
            for lu, modalities in data.iterate():
                for i, (modality, modality_name) in enumerate(zip(modalities, modality_names)):
                    for column_name, column_data in modality.transpose().iterrows():
                        for feature_name in feature_names:
                            column_names.append('{0}_{1}_{2}'.format(modality_name, column_name, feature_name))
                    if (i == 0):
                        column_names.extend(feature_accel)
                break

        rows = []

        for ri, (lu, modalities) in enumerate(data.iterate()):
            row = []
            for i, modality in enumerate(modalities):
                for name, column_data in modality.transpose().iterrows():
                    if len(column_data) > 3:
                        row.extend(map(lambda ff: ff(column_data), feature_functions))

                    else:
                        """
                        If no data is available, put nan placeholders to keep the column widths consistent
                        """
                        row.extend([np.nan] * num_ff)
                if (i == 0):
                    row.extend(extract_accel(modality))
            assert len(row) == len(column_names)
            rows.append(row)
            # Report progress
            if train_test is 'train':
                if np.mod(ri + 1, 50) == 0:
                    print ("{:5}".format(str(ri + 1))),

            if np.mod(ri + 1, 500) == 0:
                print

        df = pd.DataFrame(rows)
        df.columns = column_names

        df.to_csv('../input/public_data/{}/{}/columns_v5.csv'.format(train_test, stub_name),
                  index=False)  # if train_test is 'train' or np.mod(fi, 50) == 0:
        if train_test is 'train': print
        print ("Finished feature extraction for {}/{}\n".format(train_test, stub_name))
