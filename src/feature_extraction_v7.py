# -*- coding: utf-8 -*-
"""
This class generate video features
"""

import warnings

import pandas as pd
import scipy.spatial.distance as distance
from spectrum import *

from visualise_data import Sequence

warnings.filterwarnings('ignore')

# We will want to keep track of the feature names for later, so we will collect these in the following list:
column_names = []
modality_names = ['acceleration', 'rssi', 'pir', 'video_living_room', 'video_kitchen', 'video_hallway']
#video
feature_video = ['width_2d','length_2d','ratio_2d'
                            ,'center_dist_2d_max','center_dist_2d_min','center_dist_2d_median','center_dist_2d_mean'
                            ,'height_3d_max','height_3d_min','height_3d_median', 'height_3d_mean'
                            ,'length_3d_max','length_3d_min','length_3d_median', 'length_3d_mean'
                            ,'width_3d_max','width_3d_min','width_3d_median', 'width_3d_mean'
                            ,'ratio_wid_len_3d_max','ratio_wid_len_3d_min','ratio_wid_len_3d_median', 'ratio_wid_len_3d_mean'
                            ,'ratio_wid_hei_3d_max','ratio_wid_hei_3d_min','ratio_wid_hei_3d_median', 'ratio_wid_hei_3d_mean'
                            ,'center_dist_3d_max','center_dist_3d_min','center_dist_3d_median','center_dist_3d_mean'
                 ]
num_ff_video = len(feature_video)

def dist(row,point1, point2):
    points = [row[point1],row[point2]]
    return distance.pdist(points)[0]

def extract_video(video):
    """video features - extracted bounding boxes
    1-The aspect ratio should indicate the pose of the person (are they standing/sitting?)
    2-centre of both the 2D and 3D bounding boxes should where the person is (are they on the sofa, for example)
    3-the trajectory of the bounding box should indicate whether the person recorded is walking or stationary
    video format:
    centre_2d_x, centre_2d_y, bb_2d_br_x, bb_2d_br_y, bb_2d_tl_x, bb_2d_tl_y,
    centre_3d_x, centre_3d_y, centre_3d_z, bb_3d_brb_x, bb_3d_brb_y, bb_3d_brb_z, bb_3d_flt_x, bb_3d_flt_y, bb_3d_flt_z
    """
    if(video.shape[0]<2): #insufficient
        return [np.nan] * num_ff_video

    ff_video = []
    #derive width, height, ratio from top left and bottom right coordinators
    min_x = np.min(video['bb_2d_tl_x'])
    max_x = np.max(video['bb_2d_br_x'])
    width_2d = abs(max_x - min_x)
    ff_video.append(width_2d)

    min_y = np.min(video['bb_2d_tl_y'])
    max_y = np.max(video['bb_2d_br_y'])
    length_2d = abs(max_y - min_y)
    ff_video.append(length_2d)
    ratio_2d = width_2d/float(length_2d)
    ff_video.append(ratio_2d)

    video['center_points_2d'] = video[['centre_2d_x', 'centre_2d_y']].apply(tuple, axis=1)#zip(video.centre_2d_x, video.centre_2d_x)
    center_points_2d = video['center_points_2d'].values.tolist()

    dist_2d = distance.pdist(center_points_2d)
    ff_video.append(dist_2d.max())
    ff_video.append(dist_2d.min())
    ff_video.append(np.median(dist_2d))
    ff_video.append(np.average(dist_2d))

    #3d
    video['height_3d'] = abs(video['bb_3d_brb_z']-video['bb_3d_flt_z'])
    arr = video['height_3d'].values
    ff_video.append(arr.max())
    ff_video.append(arr.min())
    ff_video.append(np.median(arr))
    ff_video.append(np.average(arr))

    #width, length
    video['A'] = video[['bb_3d_flt_x', 'bb_3d_flt_y','bb_3d_flt_z']].apply(tuple, axis=1)
    video['B'] = video[['centre_3d_x', 'centre_3d_y','centre_3d_z']].apply(tuple, axis=1)
    video['C'] = video[['bb_3d_brb_x', 'bb_3d_brb_y','bb_3d_brb_z']].apply(tuple, axis=1)

    video['N'] = video[['centre_3d_x', 'bb_3d_brb_y','centre_3d_z']].apply(tuple, axis=1)
    video['P'] = video[['bb_3d_brb_x', 'centre_3d_y','centre_3d_z']].apply(tuple, axis=1)

    video['dist_C_B'] = video.apply(lambda row: dist(row,'C','B'), axis=1)
    video['dist_C_A'] = video.apply(lambda row: dist(row,'C','A'), axis=1)
    video['dist_B_N'] = video.apply(lambda row: dist(row,'B','N'), axis=1)
    video['dist_B_P'] = video.apply(lambda row: dist(row,'B','P'), axis=1)

    video['k'] = video['dist_C_B']/video['dist_C_A']
    video['length_3d'] = video['dist_B_N']*video['k']
    video['width_3d'] = video['dist_B_P']*video['k']

    video['ratio_wid_len_3d'] = video['width_3d']/video['length_3d']
    video['ratio_wid_hig_3d'] = video['width_3d']/video['height_3d']

    arr = video['length_3d'].values
    ff_video.append(arr.max())
    ff_video.append(arr.min())
    ff_video.append(np.median(arr))
    ff_video.append(np.average(arr))

    arr = video['width_3d'].values
    ff_video.append(arr.max())
    ff_video.append(arr.min())
    ff_video.append(np.median(arr))
    ff_video.append(np.average(arr))

    arr = video['ratio_wid_len_3d'].values
    ff_video.append(arr.max())
    ff_video.append(arr.min())
    ff_video.append(np.median(arr))
    ff_video.append(np.average(arr))

    arr = video['ratio_wid_hig_3d'].values
    ff_video.append(arr.max())
    ff_video.append(arr.min())
    ff_video.append(np.median(arr))
    ff_video.append(np.average(arr))

    video['center_points_3d'] = video[['centre_3d_x', 'centre_3d_y', 'centre_3d_z']].apply(tuple, axis=1)#zip(video.centre_2d_x, video.centre_2d_x)
    center_points_3d = video['center_points_3d'].values.tolist()

    dist_3d = distance.pdist(center_points_3d)
    ff_video.append(dist_3d.max())
    ff_video.append(dist_3d.min())
    ff_video.append(np.median(dist_3d))
    ff_video.append(np.average(dist_3d))

    return ff_video

video_feature_functions = [extract_video]

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

                    if(i in (3,4,5)):
                        for feature_name in feature_video:
                            column_names.append('{0}_{1}'.format(i,feature_name))
                # Break here
                break

        rows = []

        for ri, (lu, modalities) in enumerate(data.iterate()):
            row = []

            for i, modality in enumerate(modalities):
                #video living, kitchen, hallway
                if(i in (3,4,5)):
                    row.extend(extract_video(modality))

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
        df.to_csv('../input/public_data/{}/{}/columns_v7.csv'.format(train_test, stub_name),
                  index=False)  # if train_test is 'train' or np.mod(fi, 50) == 0:
        if train_test is 'train': print
        print ("Finished feature extraction for {}/{}\n".format(train_test, stub_name))
