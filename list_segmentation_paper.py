# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:12:07 2019

This script takes raw data from an eye tracker and preprocesses it so machine learning
algorithms can use it to estimate cognitive load. Please take settings into account.
"""

import os
import csv
import glob
import pandas as pd
import numpy as np
import sys
import scipy
from scipy import interpolate
import statistics

########################### SETTINGS #############################
# Incorperate the made *_light_intensity.csv lists, this is only
# possible when the light_intensity.py script has been run. This is necessary
# If you want to take light intensity into account
use_light_intensity = True

# The following settings define the list segments. They are only relevant when
# use_video_segments has been set to False.
# Desired list segment length in seconds
segmentlength = 6 

# Optional cut-off start in seconds, default is 0
cut_off_start = 0

# Optional cut-off end in seconds, default is 0
cut_off_end = 0

# Don't cut excess video from both ends, but only from the start, True will override cut_from_end
cut_from_start = False

# Don't cut excess video from both ends, but only from the end
cut_from_end = False

# Format of the input csvs, csv or txt
csv_input_format = "txt"

# Format of the output csvs, csv or txt
csv_output_format = "csv"

# Frequency of the list in Hz (datapoints per second) 
frequency = 60

# Interpolation method (data where the absence of the phenomenon is depicted as 0 will always use nearest, for others use ‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’)
interpmethod = 'linear'

# Eye Data set name ('scan' will take the first available option, or set to e.g. 'Original', 'Processed')
eyedataset = 'Original'

# Gaze Data set name ('scan' will take the first available option, or set to e.g. 'Original', 'Processed')
gazedataset = 'scan'

# Of the following features select if you want to take them into account:
gaze_coordinates = True
pupil_coordinates = True
fixations = True
fixations_duration = True
saccades = True
saccades_duration = True
saccades_angle = True
Fahrdynamikdaten_accelX_VEH = False
Fahrdynamikdaten_velX_IN = False
Fahrdynamikdaten_accelY_VEH = False
Fahrdynamikdaten_velY = False
Fahrdynamikdaten_gaspedal = False
Fahrdynamikdaten_brakepedal = False
Fahrdynamikdaten_steeringWheelAngle = False
Fahrdynamikdaten_Blinker = False
Fahrdynamikdaten_Geschwindigkeit = True
Fahrdynamikdaten_Abstandhaltung = False
Fahrdynamikdaten_Spurhaltung_X = False
Fahrdynamikdaten_Spurhaltung_Y = False
Fahrdynamikdaten_Gaspedalstellung = False
Fahrdynamikdaten_Lenkmoment = False
amountblinks = True
PERCLOS_feature = True
# The following features are only relevant when incorperating light intensity:
RGB = True
RGB_0_5 = True
RGB_1 = True
RGB_1_5 = True
RGB_2 = True
RGB_2_5 = True
RGB_3_5 = True
RGB_5 = True
RGB_7_5 = True
RGB_10 = True
RGB_12_5 = True
RGB_15 = True
# This uses a Riemann sum to combine all data into one feature, only works when all RGB values are on and available
RGB_combined = True
##################################################################

# if the 'csv_output' directory is not there, create it
if not os.path.isdir("csv_output"):
    os.mkdir("csv_output")

# finding all video files of the set format in folder and adding them to a csv list
path = os.path.dirname(__file__) # get current path
os.chdir(path) # change path to folder of .py file
with open('csv_output/InputCSVList.'+csv_output_format, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['Filename','Index'])
csvFile.close()

# adding all txt files in csv_input folder to CSVList.csv and indexing them
i = 0
SegmentID = 0
devicelist = []
allheaders = []
for file in glob.glob("csv_input/*_CsvData."+csv_input_format):
    row = [file,str(i)]
    with open('csv_output/InputCSVList.'+csv_output_format, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()

    # Get dataframe from csv
    if csv_input_format == "txt":
        dataframe = pd.read_csv(file, sep = '\t', low_memory = False)
    elif csv_input_format == "csv":
        dataframe = pd.read_csv(file)
    else:
        print('Please choose a valid format for the csv data')
        sys.exit()
        
    # convert time to time from first datapoint
    data_t_start = str(dataframe['rec_time'][0])
    hours, minutes, seconds = [float(x) for x in data_t_start.split(':')]
    data_t_start = 3600 * hours + 60 * minutes + seconds
    dataframe['UTC']=(dataframe['UTC']-dataframe['UTC'][0])/1000 + data_t_start
        
    columnname = [col for col in dataframe.columns if 'Left Eye_Pupil Height' in col]
        
    if 'Dikablis Professional' in columnname[0]:
        device = 'Dikablis Professional'
    elif 'Dikablis Glasses 3' in columnname[0]:
        device = 'Dikablis Glasses 3'
    else:
        print('Device not recognised. Perhaps filenames are not in the correct format.')
        break
    devicelist.append(device)
    
    if eyedataset == 'scan':
        eyedataset = columnname[0].replace(device+'_Eye Data_','')
        eyedataset = eyedataset.replace('_Left Eye_Pupil Height','')
        eyereset = True
    else:
        eyereset = False
        
    if gazedataset == 'scan':
        columnname_gaze = [col for col in dataframe.columns if '_Gaze_Gaze X' in col]
        gazedataset = columnname_gaze[0].replace(device+'_Field Data_Scene Cam_','')
        gazedataset = gazedataset.replace('_Gaze_Gaze X','')
        gazereset = True
    else:
        gazereset = False
    
    if gaze_coordinates:
        if device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze X' in dataframe.columns and device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze Y' in dataframe.columns:
            dataframe_gaze = dataframe[['UTC',device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze X',device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze Y']]
            
            # Drop NaN values
            dataframe2_gaze = dataframe_gaze[np.isfinite(dataframe_gaze[device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze X'])]
            # Get the time to previous and the next data point
            time_to_last = dataframe2_gaze['UTC']-dataframe2_gaze['UTC'].shift(1)
            time_to_next = dataframe2_gaze['UTC'].shift(-1)-dataframe2_gaze['UTC']
            # Calculate the time gap that would be there if the point was omitted
            gap_if_point_omitted = time_to_last + time_to_next
            
            # See if time to previous or next data point is lower than 10 ms (should not be possible)
            low_time_to_last = time_to_last < 0.01
            low_time_to_next = time_to_next < 0.01
            # Check if time between surrounding points is lower than 25 ms (should not be possible)
            low_gap_if_point_omitted = gap_if_point_omitted < 0.025
            # See if zero
            checkcolumn = dataframe2_gaze[device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze X']
            is_zero = checkcolumn.isin([0])
            
            # Determine fake zeros by the hand of 3 different strategies
            fake_zero_last = is_zero & low_time_to_last
            fake_zero_next = is_zero & low_time_to_next
            fake_zero_omitted = is_zero & low_gap_if_point_omitted 
            
            # Flip to get real values
            real_values_last = [not i for i in fake_zero_last]
            real_values_next = [not i for i in fake_zero_next]
            real_values_omitted = [not i for i in fake_zero_omitted]
        
            # Filter the data
            dataframe_last_gaze = dataframe2_gaze[real_values_last]
            dataframe_next_gaze = dataframe2_gaze[real_values_next]
            dataframe_omitted_gaze = dataframe2_gaze[real_values_omitted]
                
            # Calculate the pauses (more than 100ms) in the recording
            pauses = pd.DataFrame()
            pauses['Time to last'] = time_to_last[time_to_last > 0.1]
            total_pause = float(np.sum(pauses))
        
            # Count the number of datapoints there should be, which is the last recording time - the total time of pausing,
            # times the frequency, plus the amount of pauses because the time between the points where a pause took place is not 0
            length_gaze = np.ceil((dataframe2_gaze['UTC'].iloc[-1] - dataframe2_gaze['UTC'].iloc[0] - total_pause) * frequency + len(pauses))
        
            markerlength_last_gaze = len(dataframe_last_gaze[device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze X'])
            markerlength_next_gaze = len(dataframe_next_gaze[device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze X'])
            markerlength_omitted_gaze = len(dataframe_omitted_gaze[device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze X'])
            
            mismatch_last_gaze = abs(markerlength_last_gaze - length_gaze)
            mismatch_next_gaze = abs(markerlength_next_gaze - length_gaze)
            mismatch_omitted_gaze = abs(markerlength_omitted_gaze - length_gaze)
            
            # Find the best strategy
            mismatch_gaze = np.min([mismatch_last_gaze,mismatch_next_gaze,mismatch_omitted_gaze])
            
            if mismatch_gaze == mismatch_last_gaze:
                dataframe3_gaze = dataframe_last_gaze
                matchingstrategy_gaze = 'last'
            elif mismatch_gaze == mismatch_next_gaze:
                dataframe3_gaze = dataframe_next_gaze
                matchingstrategy_gaze = 'next'
            elif mismatch_gaze == mismatch_omitted_gaze:
                dataframe3_gaze = dataframe_omitted_gaze
                matchingstrategy_gaze = 'omitted'
            gaze_data = True
        else:
            gaze_data = False
            print('The column \''+device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze X\' and/or the column \''+device+'_Field Data_Scene Cam_Original_Gaze_Gaze Y\' is missing from the dataset for video %i ' %(i))

    # There is no need for dropping or duplicating datapoints as all data will be interpolated to a common time vector

    # Gather and filter features for the left eye
    dataframe_lefteye = pd.DataFrame(dataframe[['UTC',device+"_Eye Data_"+eyedataset+"_Left Eye_Pupil Height",device+"_Eye Data_"+eyedataset+"_Left Eye_Pupil Width"]])
    if pupil_coordinates:
        if device+'_Eye Data_'+eyedataset+'_Left Eye_Pupil X' in dataframe.columns:
            dataframe_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Pupil X'] = dataframe[device+'_Eye Data_'+eyedataset+'_Left Eye_Pupil X']
        else:
            print('The column \''+device+'_Eye Data_'+eyedataset+'_Left Eye_Pupil X\' is missing from the dataset for video %i ' %(i))
        if device+'_Eye Data_'+eyedataset+'_Left Eye_Pupil Y' in dataframe.columns:
            dataframe_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Pupil Y'] = dataframe[device+'_Eye Data_'+eyedataset+'_Left Eye_Pupil Y']
        else:
            print('The column \''+device+'_Eye Data_'+eyedataset+'_Left Eye_Pupil Y\' is missing from the dataset for video %i ' %(i))

    if fixations:
        if device+'_Eye Data_'+eyedataset+'_Left Eye_Fixations_Fixations' in dataframe.columns:
            dataframe_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Fixations_Fixations'] = dataframe[device+'_Eye Data_'+eyedataset+'_Left Eye_Fixations_Fixations']
        else:
            print('The column \''+device+'_Eye Data_'+eyedataset+'_Left Eye_Fixations_Fixations\' is missing from the dataset for video %i ' %(i))

    if fixations_duration:
        if device+'_Eye Data_'+eyedataset+'_Left Eye_Fixations_Fixations Duration' in dataframe.columns:
            dataframe_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Fixations_Fixations Duration'] = dataframe[device+'_Eye Data_'+eyedataset+'_Left Eye_Fixations_Fixations Duration']
        else:
            print('The column \''+device+'_Eye Data_'+eyedataset+'_Left Eye_Fixations_Fixations Duration\' is missing from the dataset for video %i ' %(i))
    
    if saccades:
        if device+'_Eye Data_'+eyedataset+'_Left Eye_Saccades_Saccades' in dataframe.columns:
            dataframe_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Saccades_Saccades'] = dataframe[device+'_Eye Data_'+eyedataset+'_Left Eye_Saccades_Saccades']
        else:
            print('The column \''+device+'_Eye Data_'+eyedataset+'_Left Eye_Saccades_Saccades\' is missing from the dataset for video %i ' %(i))

    if saccades_duration:
        if device+'_Eye Data_'+eyedataset+'_Left Eye_Saccades_Saccades Duration' in dataframe.columns:
            dataframe_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Saccades_Saccades Duration'] = dataframe[device+'_Eye Data_'+eyedataset+'_Left Eye_Saccades_Saccades Duration']
        else:
            print('The column \''+device+'_Eye Data_'+eyedataset+'_Left Eye_Saccades_Saccades Duration\' is missing from the dataset for video %i ' %(i))
        
    if saccades_angle:
        if device+'_Eye Data_'+eyedataset+'_Left Eye_Saccades_Saccades Angle' in dataframe.columns:
            dataframe_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Saccades_Saccades Angle'] = dataframe[device+'_Eye Data_'+eyedataset+'_Left Eye_Saccades_Saccades Angle']
        else:
            print('The column \''+device+'_Eye Data_'+eyedataset+'_Left Eye_Saccades_Saccades Angle\' is missing from the dataset for video %i ' %(i))

    # Drop NaN values
    dataframe2_lefteye = dataframe_lefteye[np.isfinite(dataframe_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Pupil Height'])]

    # Gather and filter features for the right eye
    dataframe_righteye = pd.DataFrame(dataframe[['UTC',device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil Height',device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil Width']])
    if pupil_coordinates:
        if device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil X' in dataframe.columns:
            dataframe_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil X'] = dataframe[device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil X']
        else:
            print('The column \''+device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil X\' is missing from the dataset for video %i ' %(i))
        if device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil Y' in dataframe.columns:
            dataframe_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil Y'] = dataframe[device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil Y']
        else:
            print('The column \''+device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil Y\' is missing from the dataset for video %i ' %(i))

    if fixations:
        if device+'_Eye Data_'+eyedataset+'_Right Eye_Fixations_Fixations' in dataframe.columns:
            dataframe_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Fixations_Fixations'] = dataframe[device+'_Eye Data_'+eyedataset+'_Right Eye_Fixations_Fixations']
        else:
            print('The column \''+device+'_Eye Data_'+eyedataset+'_Right Eye_Fixations_Fixations\' is missing from the dataset for video %i ' %(i))

    if fixations_duration:
        if device+'_Eye Data_'+eyedataset+'_Right Eye_Fixations_Fixations Duration' in dataframe.columns:
            dataframe_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Fixations_Fixations Duration'] = dataframe[device+'_Eye Data_'+eyedataset+'_Right Eye_Fixations_Fixations Duration']
        else:
            print('The column \''+device+'_Eye Data_'+eyedataset+'_Right Eye_Fixations_Fixations Duration\' is missing from the dataset for video %i ' %(i))
    
    if saccades:
        if device+'_Eye Data_'+eyedataset+'_Right Eye_Saccades_Saccades' in dataframe.columns:
            dataframe_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Saccades_Saccades'] = dataframe[device+'_Eye Data_'+eyedataset+'_Right Eye_Saccades_Saccades']
        else:
            print('The column \''+device+'_Eye Data_'+eyedataset+'_Right Eye_Saccades_Saccades\' is missing from the dataset for video %i ' %(i))

    if saccades_duration:
        if device+'_Eye Data_'+eyedataset+'_Right Eye_Saccades_Saccades Duration' in dataframe.columns:
            dataframe_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Saccades_Saccades Duration'] = dataframe[device+'_Eye Data_'+eyedataset+'_Right Eye_Saccades_Saccades Duration']
        else:
            print('The column \''+device+'_Eye Data_'+eyedataset+'_Right Eye_Saccades_Saccades Duration\' is missing from the dataset for video %i ' %(i))
        
    if saccades_angle:
        if device+'_Eye Data_'+eyedataset+'_Right Eye_Saccades_Saccades Angle' in dataframe.columns:
            dataframe_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Saccades_Saccades Angle'] = dataframe[device+'_Eye Data_'+eyedataset+'_Right Eye_Saccades_Saccades Angle']
        else:
            print('The column \''+device+'_Eye Data_'+eyedataset+'_Right Eye_Saccades_Saccades Angle\' is missing from the dataset for video %i ' %(i))

    # Drop NaN values
    dataframe2_righteye = dataframe_righteye[np.isfinite(dataframe_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil Height'])]

    # Create the dataframes where blinks, 1 datapoint before and 4 datapoints after the blink are excluded
    dataframe_lefteye_corrected = pd.DataFrame(dataframe2_lefteye['UTC'])
    dataframe_lefteye_corrected['Left Pupil Diameter Corrected'] = dataframe2_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Pupil Height']
    dataframe_lefteye_corrected['Left Pupil X Corrected'] = dataframe2_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Pupil X']
    dataframe_lefteye_corrected['Left Pupil Y Corrected'] = dataframe2_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Pupil Y']
    interpdiam0 = dataframe_lefteye_corrected['Left Pupil Diameter Corrected'].shift(-1) == 0
    interpdiam1 = dataframe_lefteye_corrected['Left Pupil Diameter Corrected'] == 0
    interpdiam2 = dataframe_lefteye_corrected['Left Pupil Diameter Corrected'].shift(1) == 0
    interpdiam3 = dataframe_lefteye_corrected['Left Pupil Diameter Corrected'].shift(2) == 0
    interpdiam4 = dataframe_lefteye_corrected['Left Pupil Diameter Corrected'].shift(3) == 0
    interpdiam5 = dataframe_lefteye_corrected['Left Pupil Diameter Corrected'].shift(4) == 0
    interpdiam = interpdiam0 | interpdiam1 | interpdiam2 | interpdiam3 | interpdiam4 | interpdiam5
    interpdiam = [not t for t in interpdiam]
    dataframe_lefteye_corrected = dataframe_lefteye_corrected[interpdiam]

    dataframe_righteye_corrected = pd.DataFrame(dataframe2_righteye['UTC'])
    dataframe_righteye_corrected['Right Pupil Diameter Corrected'] = dataframe2_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil Height']
    dataframe_righteye_corrected['Right Pupil X Corrected'] = dataframe2_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil X']
    dataframe_righteye_corrected['Right Pupil Y Corrected'] = dataframe2_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil Y']
    interpdiam0 = dataframe_righteye_corrected['Right Pupil Diameter Corrected'].shift(-1) == 0
    interpdiam1 = dataframe_righteye_corrected['Right Pupil Diameter Corrected'] == 0
    interpdiam2 = dataframe_righteye_corrected['Right Pupil Diameter Corrected'].shift(1) == 0
    interpdiam3 = dataframe_righteye_corrected['Right Pupil Diameter Corrected'].shift(2) == 0
    interpdiam4 = dataframe_righteye_corrected['Right Pupil Diameter Corrected'].shift(3) == 0
    interpdiam5 = dataframe_righteye_corrected['Right Pupil Diameter Corrected'].shift(4) == 0
    interpdiam = interpdiam0 | interpdiam1 | interpdiam2 | interpdiam3 | interpdiam4 | interpdiam5
    interpdiam = [not t for t in interpdiam]
    dataframe_righteye_corrected = dataframe_righteye_corrected[interpdiam]

    dataframe_gaze_corrected = pd.DataFrame(dataframe3_gaze['UTC'])
    dataframe_gaze_corrected['Gaze X Corrected'] = dataframe3_gaze[device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze X']
    dataframe_gaze_corrected['Gaze Y Corrected'] = dataframe3_gaze[device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze Y']
    interpdiam0 = dataframe_gaze_corrected['Gaze X Corrected'].shift(-1) == 0
    interpdiam1 = dataframe_gaze_corrected['Gaze X Corrected'] == 0
    interpdiam2 = dataframe_gaze_corrected['Gaze X Corrected'].shift(1) == 0
    interpdiam3 = dataframe_gaze_corrected['Gaze X Corrected'].shift(2) == 0
    interpdiam4 = dataframe_gaze_corrected['Gaze X Corrected'].shift(3) == 0
    interpdiam5 = dataframe_gaze_corrected['Gaze X Corrected'].shift(4) == 0
    interpdiam = interpdiam0 | interpdiam1 | interpdiam2 | interpdiam3 | interpdiam4 | interpdiam5
    interpdiam = [not t for t in interpdiam]
    dataframe_gaze_corrected = dataframe_gaze_corrected[interpdiam]

    # Gather and filter features for the driving simulator data
    dataframe_driving = pd.DataFrame(dataframe['UTC'])
    
    if 'nandeleter' in locals():
        del nandeleter
    
    if Fahrdynamikdaten_accelX_VEH:
        if 'Fahrdynamikdaten_accelX_VEH' in dataframe.columns:
            dataframe_driving['Fahrdynamikdaten_accelX_VEH'] = dataframe['Fahrdynamikdaten_accelX_VEH']
            nandeleter = 'Fahrdynamikdaten_accelX_VEH'
        else:
            print('The column \'Fahrdynamikdaten_accelX_VEH\' is missing from the dataset for video %i ' %(i))
            
    if Fahrdynamikdaten_velX_IN:
        if 'Fahrdynamikdaten_velX_IN' in dataframe.columns:
            dataframe_driving['Fahrdynamikdaten_velX_IN'] = dataframe['Fahrdynamikdaten_velX_IN']
            nandeleter = 'Fahrdynamikdaten_velX_IN'
        else:
            print('The column \'Fahrdynamikdaten_velX_IN\' is missing from the dataset for video %i ' %(i))
            
    if Fahrdynamikdaten_accelY_VEH:
        if 'Fahrdynamikdaten_accelY_VEH' in dataframe.columns:
            dataframe_driving['Fahrdynamikdaten_accelY_VEH'] = dataframe['Fahrdynamikdaten_accelY_VEH']
            nandeleter = 'Fahrdynamikdaten_accelY_VEH'
        else:
            print('The column \'Fahrdynamikdaten_accelY_VEH\' is missing from the dataset for video %i ' %(i))

    if Fahrdynamikdaten_velY:
        if 'Fahrdynamikdaten_velY' in dataframe.columns:
            dataframe_driving['Fahrdynamikdaten_velY'] = dataframe['Fahrdynamikdaten_velY']
            nandeleter = 'Fahrdynamikdaten_velY'
        else:
            print('The column \'Fahrdynamikdaten_velY\' is missing from the dataset for video %i ' %(i))

    if Fahrdynamikdaten_gaspedal:
        if 'Fahrdynamikdaten_gaspedal' in dataframe.columns:
            dataframe_driving['Fahrdynamikdaten_gaspedal'] = dataframe['Fahrdynamikdaten_gaspedal']
            nandeleter = 'Fahrdynamikdaten_gaspedal'
        else:
            print('The column \'Fahrdynamikdaten_gaspedal\' is missing from the dataset for video %i ' %(i))
  
    if Fahrdynamikdaten_brakepedal:
        if 'Fahrdynamikdaten_brakepedal' in dataframe.columns:
            dataframe_driving['Fahrdynamikdaten_brakepedal'] = dataframe['Fahrdynamikdaten_brakepedal']
            nandeleter = 'Fahrdynamikdaten_brakepedal'
        else:
            print('The column \'Fahrdynamikdaten_brakepedal\' is missing from the dataset for video %i ' %(i))
   
    if Fahrdynamikdaten_steeringWheelAngle:
        if 'Fahrdynamikdaten_steeringWheelAngle' in dataframe.columns:
            dataframe_driving['Fahrdynamikdaten_steeringWheelAngle'] = dataframe['Fahrdynamikdaten_steeringWheelAngle']
            nandeleter = 'Fahrdynamikdaten_steeringWheelAngle'
        else:
            print('The column \'Fahrdynamikdaten_steeringWheelAngle\' is missing from the dataset for video %i ' %(i))
    
    if Fahrdynamikdaten_Blinker:
        if 'Fahrdynamikdaten_Blinker' in dataframe.columns:
            dataframe_driving['Fahrdynamikdaten_Blinker'] = dataframe['Fahrdynamikdaten_Blinker']
            nandeleter = 'Fahrdynamikdaten_Blinker'
        else:
            print('The column \'Fahrdynamikdaten_Blinker\' is missing from the dataset for video %i ' %(i))
            
    if Fahrdynamikdaten_Geschwindigkeit:
        if 'Fahrdynamikdaten_Geschwindigkeit' in dataframe.columns:
            dataframe_driving['Fahrdynamikdaten_Geschwindigkeit'] = dataframe['Fahrdynamikdaten_Geschwindigkeit']
            nandeleter = 'Fahrdynamikdaten_Geschwindigkeit'
        else:
            print('The column \'Fahrdynamikdaten_Geschwindigkeit\' is missing from the dataset for video %i ' %(i))

    if Fahrdynamikdaten_Abstandhaltung:
        if 'Fahrdynamikdaten_Abstandhaltung' in dataframe.columns:
            dataframe_driving['Fahrdynamikdaten_Abstandhaltung'] = dataframe['Fahrdynamikdaten_Abstandhaltung']
            nandeleter = 'Fahrdynamikdaten_Abstandhaltung'
        else:
            print('The column \'Fahrdynamikdaten_Abstandhaltung\' is missing from the dataset for video %i ' %(i))
 
    if Fahrdynamikdaten_Spurhaltung_X:
        if 'Fahrdynamikdaten_Spurhaltung_X' in dataframe.columns:
            dataframe_driving['Fahrdynamikdaten_Spurhaltung_X'] = dataframe['Fahrdynamikdaten_Spurhaltung_X']
            nandeleter = 'Fahrdynamikdaten_Spurhaltung_X'
        else:
            print('The column \'Fahrdynamikdaten_Spurhaltung_X\' is missing from the dataset for video %i ' %(i))

    if Fahrdynamikdaten_Spurhaltung_Y:
        if 'Fahrdynamikdaten_Spurhaltung_Y' in dataframe.columns:
            dataframe_driving['Fahrdynamikdaten_Spurhaltung_Y'] = dataframe['Fahrdynamikdaten_Spurhaltung_Y']
            nandeleter = 'Fahrdynamikdaten_Spurhaltung_Y'
        else:
            print('The column \'Fahrdynamikdaten_Spurhaltung_Y\' is missing from the dataset for video %i ' %(i))

    if Fahrdynamikdaten_Gaspedalstellung:
        if 'Fahrdynamikdaten_Gaspedalstellung' in dataframe.columns:
            dataframe_driving['Fahrdynamikdaten_Gaspedalstellung'] = dataframe['Fahrdynamikdaten_Gaspedalstellung']
            nandeleter = 'Fahrdynamikdaten_Gaspedalstellung'
        else:
            print('The column \'Fahrdynamikdaten_Gaspedalstellung\' is missing from the dataset for video %i ' %(i))

    if Fahrdynamikdaten_Lenkmoment:
        if 'Fahrdynamikdaten_Lenkmoment' in dataframe.columns:
            dataframe_driving['Fahrdynamikdaten_Lenkmoment'] = dataframe['Fahrdynamikdaten_Lenkmoment']
            nandeleter = 'Fahrdynamikdaten_Lenkmoment'
        else:
            print('The column \'Fahrdynamikdaten_Lenkmoment\' is missing from the dataset for video %i ' %(i))
    
    # Drop NaN values, check if there are Fahrdynamikdaten selected
    if 'nandeleter' in locals():
        dataframe2_driving = dataframe_driving[np.isfinite(dataframe_driving[nandeleter])]
        Fahrdynamikdaten = True
    else:
        Fahrdynamikdaten = False
        
    if use_light_intensity:
        lightcsv = file.replace('CsvData.'+csv_input_format,'light_intensity.csv')
        dataframe_light = pd.read_csv(lightcsv, encoding = 'ISO-8859-1')
        
    # Get the common time vectors
    
    t_vectors = [] # nested list (list of lists) of the time vectors
    t_cuts = [] # cut off points, if there are no pauses in the recording this will just be the start and end point
    t_cuts.append(dataframe3_gaze['UTC'].iloc[0])
    
    # Add cut off points for pauses, if any
    for j in range(len(pauses)):
        t_1 = dataframe['UTC'].iloc[pauses.index[j]] - pauses['Time to last'].iloc[j]
        t_2 = dataframe['UTC'].iloc[pauses.index[j]]
        t_cuts.append(t_1)
        t_cuts.append(t_2)
    t_cuts.append(dataframe['UTC'].iloc[-1])
    
    for j in range(int(len(t_cuts)/2)):
        t_start = t_cuts[2*j]
        t_end = t_cuts[2*j+1]
        
        # calculate how many segments you can cut out
        segments_amount = int((t_end - t_start - cut_off_start - cut_off_end) / segmentlength)
        
        # print warning if the result will be 0 segments
        if segments_amount < 1:
            print('The current settings result in 0 segments for video %i' %(i))
        
        elif cut_from_end == False and cut_from_start == False:
        
            start_time = ((t_end -t_start - segments_amount * segmentlength + cut_off_start - cut_off_end) / 2)
            
        elif cut_from_start == True:
            
            start_time = t_end - t_start - segments_amount * segmentlength
            
        elif cut_from_end == True:
            
            start_time = cut_off_start 

        for k in range(segments_amount):
            t_vector_segment = list(np.linspace(start_time + k*segmentlength + t_start, start_time + (k+1)*segmentlength + t_start, segmentlength * frequency, endpoint = False))
            t_vectors.append(t_vector_segment)
    
    # get interpolation functions, unfortunately feature by feature as the function can only handle 1D data    
    if gaze_coordinates and gaze_data:
        gaze_x_func = scipy.interpolate.interp1d(dataframe3_gaze['UTC'],dataframe3_gaze[device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze X'],kind='nearest', fill_value='extrapolate')
        gaze_y_func = scipy.interpolate.interp1d(dataframe3_gaze['UTC'],dataframe3_gaze[device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze Y'],kind='nearest', fill_value='extrapolate')
        gaze_x_corrected_func = scipy.interpolate.interp1d(dataframe_gaze_corrected['UTC'],dataframe_gaze_corrected['Gaze X Corrected'],kind=interpmethod, fill_value='extrapolate')
        gaze_y_corrected_func = scipy.interpolate.interp1d(dataframe_gaze_corrected['UTC'],dataframe_gaze_corrected['Gaze Y Corrected'],kind=interpmethod, fill_value='extrapolate')
    
    if pupil_coordinates and device+'_Eye Data_'+eyedataset+'_Left Eye_Pupil X' in dataframe.columns and device+'_Eye Data_'+eyedataset+'_Left Eye_Pupil Y' in dataframe.columns and device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil X' in dataframe.columns and device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil Y' in dataframe.columns:
        left_x_func = scipy.interpolate.interp1d(dataframe2_lefteye['UTC'],dataframe2_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Pupil X'],kind='nearest', fill_value='extrapolate')
        left_y_func = scipy.interpolate.interp1d(dataframe2_lefteye['UTC'],dataframe2_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Pupil Y'],kind='nearest', fill_value='extrapolate')
        right_x_func = scipy.interpolate.interp1d(dataframe2_righteye['UTC'],dataframe2_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil X'],kind='nearest', fill_value='extrapolate')
        right_y_func = scipy.interpolate.interp1d(dataframe2_righteye['UTC'],dataframe2_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil Y'],kind='nearest', fill_value='extrapolate')
        left_x_corrected_func = scipy.interpolate.interp1d(dataframe_lefteye_corrected['UTC'],dataframe_lefteye_corrected['Left Pupil X Corrected'],kind=interpmethod, fill_value='extrapolate')
        left_y_corrected_func = scipy.interpolate.interp1d(dataframe_lefteye_corrected['UTC'],dataframe_lefteye_corrected['Left Pupil Y Corrected'],kind=interpmethod, fill_value='extrapolate')
        right_x_corrected_func = scipy.interpolate.interp1d(dataframe_righteye_corrected['UTC'],dataframe_righteye_corrected['Right Pupil X Corrected'],kind=interpmethod, fill_value='extrapolate')
        right_y_corrected_func = scipy.interpolate.interp1d(dataframe_righteye_corrected['UTC'],dataframe_righteye_corrected['Right Pupil Y Corrected'],kind=interpmethod, fill_value='extrapolate')

        pupil_data = True
    else:
        pupil_data = False
        
    left_height_func = scipy.interpolate.interp1d(dataframe2_lefteye['UTC'],dataframe2_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Pupil Height'],kind='nearest', fill_value='extrapolate')
    left_width_func = scipy.interpolate.interp1d(dataframe2_lefteye['UTC'],dataframe2_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Pupil Width'],kind='nearest', fill_value='extrapolate')
    right_height_func = scipy.interpolate.interp1d(dataframe2_righteye['UTC'],dataframe2_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil Height'],kind='nearest', fill_value='extrapolate')
    right_width_func = scipy.interpolate.interp1d(dataframe2_righteye['UTC'],dataframe2_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Pupil Width'],kind='nearest', fill_value='extrapolate')

    left_diam_func = scipy.interpolate.interp1d(dataframe_lefteye_corrected['UTC'],dataframe_lefteye_corrected['Left Pupil Diameter Corrected'],kind=interpmethod, fill_value='extrapolate')
    right_diam_func = scipy.interpolate.interp1d(dataframe_righteye_corrected['UTC'],dataframe_righteye_corrected['Right Pupil Diameter Corrected'],kind=interpmethod, fill_value='extrapolate')

    if fixations and device+'_Eye Data_'+eyedataset+'_Left Eye_Fixations_Fixations' in dataframe.columns and device+'_Eye Data_'+eyedataset+'_Right Eye_Fixations_Fixations' in dataframe.columns:
        left_fix_func = scipy.interpolate.interp1d(dataframe2_lefteye['UTC'],dataframe2_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Fixations_Fixations'],kind='nearest', fill_value='extrapolate')
        right_fix_func = scipy.interpolate.interp1d(dataframe2_righteye['UTC'],dataframe2_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Fixations_Fixations'],kind='nearest', fill_value='extrapolate')
        fixations_data = True
    else:
        fixations_data = False
        
    if fixations_duration and device+'_Eye Data_'+eyedataset+'_Left Eye_Fixations_Fixations Duration' in dataframe.columns and device+'_Eye Data_'+eyedataset+'_Right Eye_Fixations_Fixations Duration' in dataframe.columns:
        left_fix_dur_func = scipy.interpolate.interp1d(dataframe2_lefteye['UTC'],dataframe2_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Fixations_Fixations Duration'],kind='nearest', fill_value='extrapolate')
        right_fix_dur_func = scipy.interpolate.interp1d(dataframe2_righteye['UTC'],dataframe2_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Fixations_Fixations Duration'],kind='nearest', fill_value='extrapolate')
        fixations_duration_data = True
    else:
        fixations_duration_data = False

    if saccades and device+'_Eye Data_'+eyedataset+'_Left Eye_Saccades_Saccades' in dataframe.columns and device+'_Eye Data_'+eyedataset+'_Right Eye_Saccades_Saccades' in dataframe.columns:
        left_sacc_func = scipy.interpolate.interp1d(dataframe2_lefteye['UTC'],dataframe2_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Saccades_Saccades'],kind='nearest', fill_value='extrapolate')
        right_sacc_func = scipy.interpolate.interp1d(dataframe2_righteye['UTC'],dataframe2_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Saccades_Saccades'],kind='nearest', fill_value='extrapolate')
        saccades_data = True
    else:
        saccades_data = False
    
    if saccades_duration and device+'_Eye Data_'+eyedataset+'_Left Eye_Saccades_Saccades Duration' in dataframe.columns and device+'_Eye Data_'+eyedataset+'_Right Eye_Saccades_Saccades Duration' in dataframe.columns:
        left_sacc_dur_func = scipy.interpolate.interp1d(dataframe2_lefteye['UTC'],dataframe2_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Saccades_Saccades Duration'],kind='nearest', fill_value='extrapolate')
        right_sacc_dur_func = scipy.interpolate.interp1d(dataframe2_righteye['UTC'],dataframe2_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Saccades_Saccades Duration'],kind='nearest', fill_value='extrapolate')
        saccades_duration_data = True
    else:
        saccades_duration_data = False

    if saccades_angle and device+'_Eye Data_'+eyedataset+'_Left Eye_Saccades_Saccades Angle' in dataframe.columns and device+'_Eye Data_'+eyedataset+'_Right Eye_Saccades_Saccades Angle' in dataframe.columns:
        left_sacc_ang_func = scipy.interpolate.interp1d(dataframe2_lefteye['UTC'],dataframe2_lefteye[device+'_Eye Data_'+eyedataset+'_Left Eye_Saccades_Saccades Angle'],kind='nearest', fill_value='extrapolate')
        right_sacc_ang_func = scipy.interpolate.interp1d(dataframe2_righteye['UTC'],dataframe2_righteye[device+'_Eye Data_'+eyedataset+'_Right Eye_Saccades_Saccades Angle'],kind='nearest', fill_value='extrapolate')
        saccades_angle_data = True
    else:
        saccades_angle_data = False
    
    if Fahrdynamikdaten_accelX_VEH and 'Fahrdynamikdaten_accelX_VEH' in dataframe.columns:
        accelX_VEH_func = scipy.interpolate.interp1d(dataframe2_driving['UTC'],dataframe2_driving['Fahrdynamikdaten_accelX_VEH'],kind=interpmethod, fill_value='extrapolate')
        Fahrdynamikdaten_accelX_VEH_data = True
    else:
        Fahrdynamikdaten_accelX_VEH_data = False

    if Fahrdynamikdaten_velX_IN and 'Fahrdynamikdaten_velX_IN' in dataframe.columns:
        velX_IN_func = scipy.interpolate.interp1d(dataframe2_driving['UTC'],dataframe2_driving['Fahrdynamikdaten_velX_IN'],kind=interpmethod, fill_value='extrapolate')
        Fahrdynamikdaten_velX_IN_data = True
    else:
        Fahrdynamikdaten_velX_IN_data = False
    
    if Fahrdynamikdaten_accelY_VEH and 'Fahrdynamikdaten_accelY_VEH' in dataframe.columns:
        accelY_VEH_func = scipy.interpolate.interp1d(dataframe2_driving['UTC'],dataframe2_driving['Fahrdynamikdaten_accelY_VEH'],kind=interpmethod, fill_value='extrapolate')
        Fahrdynamikdaten_accelY_VEH_data = True
    else:
        Fahrdynamikdaten_accelY_VEH_data = False

    if Fahrdynamikdaten_velY and 'Fahrdynamikdaten_velY' in dataframe.columns:
        velY_func = scipy.interpolate.interp1d(dataframe2_driving['UTC'],dataframe2_driving['Fahrdynamikdaten_velY'],kind=interpmethod, fill_value='extrapolate')
        Fahrdynamikdaten_velY_data = True
    else:
        Fahrdynamikdaten_velY_data = False

    if Fahrdynamikdaten_gaspedal and 'Fahrdynamikdaten_gaspedal' in dataframe.columns:
        gaspedal_func = scipy.interpolate.interp1d(dataframe2_driving['UTC'],dataframe2_driving['Fahrdynamikdaten_gaspedal'],kind=interpmethod, fill_value='extrapolate')
        Fahrdynamikdaten_gaspedal_data = True
    else:
        Fahrdynamikdaten_gaspedal_data = False
   
    if Fahrdynamikdaten_brakepedal and 'Fahrdynamikdaten_brakepedal' in dataframe.columns:
        brakepedal_func = scipy.interpolate.interp1d(dataframe2_driving['UTC'],dataframe2_driving['Fahrdynamikdaten_brakepedal'],kind=interpmethod, fill_value='extrapolate')
        Fahrdynamikdaten_brakepedal_data = True
    else:
        Fahrdynamikdaten_brakepedal_data = False
   
    if Fahrdynamikdaten_steeringWheelAngle and 'Fahrdynamikdaten_steeringWheelAngle' in dataframe.columns:
        steeringWheelAngle_func = scipy.interpolate.interp1d(dataframe2_driving['UTC'],dataframe2_driving['Fahrdynamikdaten_steeringWheelAngle'],kind=interpmethod, fill_value='extrapolate')
        Fahrdynamikdaten_steeringWheelAngle_data = True
    else:
        Fahrdynamikdaten_steeringWheelAngle_data = False
  
    if Fahrdynamikdaten_Blinker and 'Fahrdynamikdaten_Blinker' in dataframe.columns:
        Blinker_func = scipy.interpolate.interp1d(dataframe2_driving['UTC'],dataframe2_driving['Fahrdynamikdaten_Blinker'],kind='nearest', fill_value='extrapolate')
        Fahrdynamikdaten_Blinker_data = True
    else:
        Fahrdynamikdaten_Blinker_data = False

    if Fahrdynamikdaten_Geschwindigkeit and 'Fahrdynamikdaten_Geschwindigkeit' in dataframe.columns:
        Geschwindigkeit_func = scipy.interpolate.interp1d(dataframe2_driving['UTC'],dataframe2_driving['Fahrdynamikdaten_Geschwindigkeit'],kind=interpmethod, fill_value='extrapolate')
        Fahrdynamikdaten_Geschwindigkeit_data = True
    else:
        Fahrdynamikdaten_Geschwindigkeit_data = False

    if Fahrdynamikdaten_Abstandhaltung and 'Fahrdynamikdaten_Abstandhaltung' in dataframe.columns:
        Abstandhaltung_func = scipy.interpolate.interp1d(dataframe2_driving['UTC'],dataframe2_driving['Fahrdynamikdaten_Abstandhaltung'],kind=interpmethod, fill_value='extrapolate')
        Fahrdynamikdaten_Abstandhaltung_data = True
    else:
        Fahrdynamikdaten_Abstandhaltung_data = False

    if Fahrdynamikdaten_Spurhaltung_X and 'Fahrdynamikdaten_Spurhaltung_X' in dataframe.columns:
        Spurhaltung_X_func = scipy.interpolate.interp1d(dataframe2_driving['UTC'],dataframe2_driving['Fahrdynamikdaten_Spurhaltung_X'],kind=interpmethod, fill_value='extrapolate')
        Fahrdynamikdaten_Spurhaltung_X_data = True
    else:
        Fahrdynamikdaten_Spurhaltung_X_data = False

    if Fahrdynamikdaten_Spurhaltung_Y and 'Fahrdynamikdaten_Spurhaltung_Y' in dataframe.columns:
        Spurhaltung_Y_func = scipy.interpolate.interp1d(dataframe2_driving['UTC'],dataframe2_driving['Fahrdynamikdaten_Spurhaltung_Y'],kind=interpmethod, fill_value='extrapolate')
        Fahrdynamikdaten_Spurhaltung_Y_data = True
    else:
        Fahrdynamikdaten_Spurhaltung_Y_data = False

    if Fahrdynamikdaten_Gaspedalstellung and 'Fahrdynamikdaten_Gaspedalstellung' in dataframe.columns:
        Gaspedalstellung_func = scipy.interpolate.interp1d(dataframe2_driving['UTC'],dataframe2_driving['Fahrdynamikdaten_Gaspedalstellung'],kind=interpmethod, fill_value='extrapolate')
        Fahrdynamikdaten_Gaspedalstellung_data = True
    else:
        Fahrdynamikdaten_Gaspedalstellung_data = False

    if Fahrdynamikdaten_Lenkmoment and 'Fahrdynamikdaten_Lenkmoment' in dataframe.columns:
        Lenkmoment_func = scipy.interpolate.interp1d(dataframe2_driving['UTC'],dataframe2_driving['Fahrdynamikdaten_Lenkmoment'],kind=interpmethod, fill_value='extrapolate')
        Fahrdynamikdaten_Lenkmoment_data = True
    else:
        Fahrdynamikdaten_Lenkmoment_data = False
        
    if RGB and 'R (RGB)' in dataframe_light.columns:
        R_RGB_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['R (RGB)'],kind='nearest', fill_value='extrapolate')
        G_RGB_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['G (RGB)'],kind='nearest', fill_value='extrapolate')
        B_RGB_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['B (RGB)'],kind='nearest', fill_value='extrapolate')
        RGB_data = True
    else:
        RGB_data = False
        
    if RGB_0_5 and 'R (0.5)' in dataframe_light.columns:
        R_RGB_0_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['R (0.5)'],kind='nearest', fill_value='extrapolate')
        G_RGB_0_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['G (0.5)'],kind='nearest', fill_value='extrapolate')
        B_RGB_0_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['B (0.5)'],kind='nearest', fill_value='extrapolate')
        RGB_0_5_data = True
    else:
        RGB_0_5_data = False
    
    if RGB_1 and 'R (1)' in dataframe_light.columns:
        R_RGB_1_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['R (1)'],kind='nearest', fill_value='extrapolate')
        G_RGB_1_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['G (1)'],kind='nearest', fill_value='extrapolate')
        B_RGB_1_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['B (1)'],kind='nearest', fill_value='extrapolate')
        RGB_1_data = True
    else:
        RGB_1_data = False
        
    if RGB_1_5 and 'R (1.5)' in dataframe_light.columns:
        R_RGB_1_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['R (1.5)'],kind='nearest', fill_value='extrapolate')
        G_RGB_1_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['G (1.5)'],kind='nearest', fill_value='extrapolate')
        B_RGB_1_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['B (1.5)'],kind='nearest', fill_value='extrapolate')
        RGB_1_5_data = True
    else:
        RGB_1_5_data = False
    
    if RGB_2 and 'R (2)' in dataframe_light.columns:
        R_RGB_2_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['R (2)'],kind='nearest', fill_value='extrapolate')
        G_RGB_2_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['G (2)'],kind='nearest', fill_value='extrapolate')
        B_RGB_2_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['B (2)'],kind='nearest', fill_value='extrapolate')
        RGB_2_data = True
    else:
        RGB_2_data = False
        
    if RGB_2_5 and 'R (2.5)' in dataframe_light.columns:
        R_RGB_2_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['R (2.5)'],kind='nearest', fill_value='extrapolate')
        G_RGB_2_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['G (2.5)'],kind='nearest', fill_value='extrapolate')
        B_RGB_2_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['B (2.5)'],kind='nearest', fill_value='extrapolate')
        RGB_2_5_data = True
    else:
        RGB_2_5_data = False
     
    if RGB_3_5 and 'R (3.5)' in dataframe_light.columns:
        R_RGB_3_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['R (3.5)'],kind='nearest', fill_value='extrapolate')
        G_RGB_3_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['G (3.5)'],kind='nearest', fill_value='extrapolate')
        B_RGB_3_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['B (3.5)'],kind='nearest', fill_value='extrapolate')
        RGB_3_5_data = True
    else:
        RGB_3_5_data = False
        
    if RGB_5 and 'R (5)' in dataframe_light.columns:
        R_RGB_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['R (5)'],kind='nearest', fill_value='extrapolate')
        G_RGB_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['G (5)'],kind='nearest', fill_value='extrapolate')
        B_RGB_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['B (5)'],kind='nearest', fill_value='extrapolate')
        RGB_5_data = True
    else:
        RGB_5_data = False
        
    if RGB_7_5 and 'R (7.5)' in dataframe_light.columns:
        R_RGB_7_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['R (7.5)'],kind='nearest', fill_value='extrapolate')
        G_RGB_7_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['G (7.5)'],kind='nearest', fill_value='extrapolate')
        B_RGB_7_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['B (7.5)'],kind='nearest', fill_value='extrapolate')
        RGB_7_5_data = True
    else:
        RGB_7_5_data = False
        
    if RGB_10 and 'R (10)' in dataframe_light.columns:
        R_RGB_10_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['R (10)'],kind='nearest', fill_value='extrapolate')
        G_RGB_10_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['G (10)'],kind='nearest', fill_value='extrapolate')
        B_RGB_10_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['B (10)'],kind='nearest', fill_value='extrapolate')
        RGB_10_data = True
    else:
        RGB_10_data = False       
        
    if RGB_12_5 and 'R (12.5)' in dataframe_light.columns:
        R_RGB_12_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['R (12.5)'],kind='nearest', fill_value='extrapolate')
        G_RGB_12_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['G (12.5)'],kind='nearest', fill_value='extrapolate')
        B_RGB_12_5_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['B (12.5)'],kind='nearest', fill_value='extrapolate')
        RGB_12_5_data = True
    else:
        RGB_12_5_data = False

    if RGB_15 and 'R (15)' in dataframe_light.columns:
        R_RGB_15_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['R (15)'],kind='nearest', fill_value='extrapolate')
        G_RGB_15_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['G (15)'],kind='nearest', fill_value='extrapolate')
        B_RGB_15_func = scipy.interpolate.interp1d(dataframe_light['Time'],dataframe_light['B (15)'],kind='nearest', fill_value='extrapolate')
        RGB_15_data = True
    else:
        RGB_15_data = False

    for j in range(len(t_vectors)):
        if j == 0:
            addheader = True
        else:
            addheader = False
        dataframeoutput = pd.DataFrame()
        dataframeoutput['Path Input']=[path+'\\'+file]*len(t_vectors[j]) 
        dataframeoutput['Filename Input'] = [file]*len(t_vectors[j]) 
        if 'Proband' in dataframeoutput['Filename Input'][0]:
            proband = dataframeoutput['Filename Input'][0][17:19]
            dataframeoutput['Proband'] = proband.replace('_','')
        dataframeoutput['Input ID'] = [i]*len(t_vectors[j]) 
        dataframeoutput['Segment in video'] = [j]*len(t_vectors[j])
        dataframeoutput['Segment ID'] = [SegmentID]*len(t_vectors[j])
        dataframeoutput['Time in Recording'] = t_vectors[j]
        if Fahrdynamikdaten_accelX_VEH_data:
            dataframeoutput['Fahrdynamikdaten_accelX_VEH'] = accelX_VEH_func(t_vectors[j])
        if Fahrdynamikdaten_velX_IN_data:
            dataframeoutput['Fahrdynamikdaten_velX_IN'] = velX_IN_func(t_vectors[j])
        if Fahrdynamikdaten_accelY_VEH_data:
            dataframeoutput['Fahrdynamikdaten_accelY_VEH'] = accelY_VEH_func(t_vectors[j])
        if Fahrdynamikdaten_velY_data:
            dataframeoutput['Fahrdynamikdaten_velY'] = velY_func(t_vectors[j])
        if Fahrdynamikdaten_gaspedal_data:
            dataframeoutput['Fahrdynamikdaten_gaspedal'] = gaspedal_func(t_vectors[j])
        if Fahrdynamikdaten_brakepedal_data:
            dataframeoutput['Fahrdynamikdaten_brakepedal'] = brakepedal_func(t_vectors[j])
        if Fahrdynamikdaten_steeringWheelAngle_data:
            dataframeoutput['Fahrdynamikdaten_steeringWheelAngle'] = steeringWheelAngle_func(t_vectors[j])
        if Fahrdynamikdaten_Blinker_data:
            dataframeoutput['Fahrdynamikdaten_Blinker'] = Blinker_func(t_vectors[j])
        if Fahrdynamikdaten_Geschwindigkeit_data:
            dataframeoutput['Fahrdynamikdaten_Geschwindigkeit'] = Geschwindigkeit_func(t_vectors[j])
        if Fahrdynamikdaten_Abstandhaltung_data:
            dataframeoutput['Fahrdynamikdaten_Abstandhaltung'] = Abstandhaltung_func(t_vectors[j])
        if Fahrdynamikdaten_Spurhaltung_X_data:
            dataframeoutput['Fahrdynamikdaten_Spurhaltung_X'] = Spurhaltung_X_func(t_vectors[j])
        if Fahrdynamikdaten_Spurhaltung_Y_data:
            dataframeoutput['Fahrdynamikdaten_Spurhaltung_Y'] = Spurhaltung_Y_func(t_vectors[j])
        if Fahrdynamikdaten_Gaspedalstellung_data:
            dataframeoutput['Fahrdynamikdaten_Gaspedalstellung'] = Gaspedalstellung_func(t_vectors[j])
        if Fahrdynamikdaten_Lenkmoment_data:
            dataframeoutput['Fahrdynamikdaten_Lenkmoment'] = Lenkmoment_func(t_vectors[j])
        if pupil_data:
            dataframeoutput['Left Eye_Pupil X'] = left_x_func(t_vectors[j])
            dataframeoutput['Left Eye_Pupil Y'] = left_y_func(t_vectors[j])
            dataframeoutput['Left Pupil X Corrected'] = left_x_corrected_func(t_vectors[j])
            dataframeoutput['Left Pupil Y Corrected'] = left_y_corrected_func(t_vectors[j])
        dataframeoutput['Left Eye_Pupil Height'] = left_height_func(t_vectors[j])
        dataframeoutput['Left Eye_Pupil Width'] = left_width_func(t_vectors[j])
        dataframeoutput['Left Eye_Pupil Diameter Corrected'] = left_diam_func(t_vectors[j])
        if fixations_data:
            dataframeoutput['Left Eye_Fixations_Fixations'] = left_fix_func(t_vectors[j])
        if fixations_duration_data:
            dataframeoutput['Left Eye_Fixations_Fixations Duration'] = left_fix_dur_func(t_vectors[j])
        if saccades_data:
            dataframeoutput['Left Eye_Saccades_Saccades'] = left_sacc_func(t_vectors[j])
        if saccades_duration_data:
            dataframeoutput['Left Eye_Saccades_Saccades Duration'] = left_sacc_dur_func(t_vectors[j])
        if saccades_angle_data:
            dataframeoutput['Left Eye_Saccades_Saccades Angle'] = left_sacc_ang_func(t_vectors[j])
        if pupil_data:
            dataframeoutput['Right Eye_Pupil X'] = right_x_func(t_vectors[j])
            dataframeoutput['Right Eye_Pupil Y'] = right_y_func(t_vectors[j])
            dataframeoutput['Right Pupil X Corrected'] = right_x_corrected_func(t_vectors[j])
            dataframeoutput['Right Pupil Y Corrected'] = right_y_corrected_func(t_vectors[j])
        dataframeoutput['Right Eye_Pupil Height'] = right_height_func(t_vectors[j])
        dataframeoutput['Right Eye_Pupil Width'] = right_width_func(t_vectors[j])
        dataframeoutput['Right Eye_Pupil Diameter Corrected'] = right_diam_func(t_vectors[j])
        dataframeoutput['Average Diameter Corrected'] = (left_diam_func(t_vectors[j]) + right_diam_func(t_vectors[j])) / 2
        dataframeoutput['Gaze X'] = gaze_x_func(t_vectors[j])
        dataframeoutput['Gaze Y'] = gaze_y_func(t_vectors[j])
        dataframeoutput['Gaze X Corrected'] = gaze_x_corrected_func(t_vectors[j])
        dataframeoutput['Gaze Y Corrected'] = gaze_y_corrected_func(t_vectors[j])
        if fixations_data:
            dataframeoutput['Right Eye_Fixations_Fixations'] = right_fix_func(t_vectors[j])
        if fixations_duration_data:
            dataframeoutput['Right Eye_Fixations_Fixations Duration'] = right_fix_dur_func(t_vectors[j])
        if saccades_data:
            dataframeoutput['Right Eye_Saccades_Saccades'] = right_sacc_func(t_vectors[j])
        if saccades_duration_data:
            dataframeoutput['Right Eye_Saccades_Saccades Duration'] = right_sacc_dur_func(t_vectors[j])
        if saccades_angle_data:
            dataframeoutput['Right Eye_Saccades_Saccades Angle'] = right_sacc_ang_func(t_vectors[j])
        if RGB_data:
            dataframeoutput['R (RGB)'] = R_RGB_func(t_vectors[j])
            dataframeoutput['G (RGB)'] = G_RGB_func(t_vectors[j])
            dataframeoutput['B (RGB)'] = B_RGB_func(t_vectors[j])
            dataframeoutput['RGB'] = 0.2989 * dataframeoutput['R (RGB)'] + 0.5871 * dataframeoutput['G (RGB)'] + 0.1140 * dataframeoutput['B (RGB)']
        if RGB_0_5_data:
            dataframeoutput['R (0.5)'] = R_RGB_0_5_func(t_vectors[j])
            dataframeoutput['G (0.5)'] = G_RGB_0_5_func(t_vectors[j])
            dataframeoutput['B (0.5)'] = B_RGB_0_5_func(t_vectors[j])
            dataframeoutput['RGB 0.5'] = 0.2989 * dataframeoutput['R (0.5)'] + 0.5871 * dataframeoutput['G (0.5)'] + 0.1140 * dataframeoutput['B (0.5)']
        if RGB_1_data:
            dataframeoutput['R (1)'] = R_RGB_1_func(t_vectors[j])
            dataframeoutput['G (1)'] = G_RGB_1_func(t_vectors[j])
            dataframeoutput['B (1)'] = B_RGB_1_func(t_vectors[j])
            dataframeoutput['RGB 1'] = 0.2989 * dataframeoutput['R (1)'] + 0.5871 * dataframeoutput['G (1)'] + 0.1140 * dataframeoutput['B (1)']
        if RGB_1_5_data:
            dataframeoutput['R (1.5)'] = R_RGB_1_5_func(t_vectors[j])
            dataframeoutput['G (1.5)'] = G_RGB_1_5_func(t_vectors[j])
            dataframeoutput['B (1.5)'] = B_RGB_1_5_func(t_vectors[j])            
            dataframeoutput['RGB 1.5'] = 0.2989 * dataframeoutput['R (1.5)'] + 0.5871 * dataframeoutput['G (1.5)'] + 0.1140 * dataframeoutput['B (1.5)']
        if RGB_2_data:
            dataframeoutput['R (2)'] = R_RGB_2_func(t_vectors[j])
            dataframeoutput['G (2)'] = G_RGB_2_func(t_vectors[j])
            dataframeoutput['B (2)'] = B_RGB_2_func(t_vectors[j])
            dataframeoutput['RGB 2'] = 0.2989 * dataframeoutput['R (2)'] + 0.5871 * dataframeoutput['G (2)'] + 0.1140 * dataframeoutput['B (2)']
        if RGB_2_5_data:
            dataframeoutput['R (2.5)'] = R_RGB_2_5_func(t_vectors[j])
            dataframeoutput['G (2.5)'] = G_RGB_2_5_func(t_vectors[j])
            dataframeoutput['B (2.5)'] = B_RGB_2_5_func(t_vectors[j])            
            dataframeoutput['RGB 2.5'] = 0.2989 * dataframeoutput['R (2.5)'] + 0.5871 * dataframeoutput['G (2.5)'] + 0.1140 * dataframeoutput['B (2.5)']
        if RGB_3_5_data:
            dataframeoutput['R (3.5)'] = R_RGB_3_5_func(t_vectors[j])
            dataframeoutput['G (3.5)'] = G_RGB_3_5_func(t_vectors[j])
            dataframeoutput['B (3.5)'] = B_RGB_3_5_func(t_vectors[j])            
            dataframeoutput['RGB 3.5'] = 0.2989 * dataframeoutput['R (3.5)'] + 0.5871 * dataframeoutput['G (3.5)'] + 0.1140 * dataframeoutput['B (3.5)']
        if RGB_5_data:
            dataframeoutput['R (5)'] = R_RGB_5_func(t_vectors[j])
            dataframeoutput['G (5)'] = G_RGB_5_func(t_vectors[j])
            dataframeoutput['B (5)'] = B_RGB_5_func(t_vectors[j])
            dataframeoutput['RGB 5'] = 0.2989 * dataframeoutput['R (5)'] + 0.5871 * dataframeoutput['G (5)'] + 0.1140 * dataframeoutput['B (5)']
        if RGB_7_5_data:
            dataframeoutput['R (7.5)'] = R_RGB_7_5_func(t_vectors[j])
            dataframeoutput['G (7.5)'] = G_RGB_7_5_func(t_vectors[j])
            dataframeoutput['B (7.5)'] = B_RGB_7_5_func(t_vectors[j])
            dataframeoutput['RGB 7.5'] = 0.2989 * dataframeoutput['R (7.5)'] + 0.5871 * dataframeoutput['G (7.5)'] + 0.1140 * dataframeoutput['B (7.5)']
        if RGB_10_data:
            dataframeoutput['R (10)'] = R_RGB_10_func(t_vectors[j])
            dataframeoutput['G (10)'] = G_RGB_10_func(t_vectors[j])
            dataframeoutput['B (10)'] = B_RGB_10_func(t_vectors[j])
            dataframeoutput['RGB 10'] = 0.2989 * dataframeoutput['R (10)'] + 0.5871 * dataframeoutput['G (10)'] + 0.1140 * dataframeoutput['B (10)']
        if RGB_12_5_data:
            dataframeoutput['R (12.5)'] = R_RGB_12_5_func(t_vectors[j])
            dataframeoutput['G (12.5)'] = G_RGB_12_5_func(t_vectors[j])
            dataframeoutput['B (12.5)'] = B_RGB_12_5_func(t_vectors[j])            
            dataframeoutput['RGB 12.5'] = 0.2989 * dataframeoutput['R (12.5)'] + 0.5871 * dataframeoutput['G (12.5)'] + 0.1140 * dataframeoutput['B (12.5)']
        if RGB_15_data:
            dataframeoutput['R (15)'] = R_RGB_15_func(t_vectors[j])
            dataframeoutput['G (15)'] = G_RGB_15_func(t_vectors[j])
            dataframeoutput['B (15)'] = B_RGB_15_func(t_vectors[j])
            dataframeoutput['RGB 15'] = 0.2989 * dataframeoutput['R (15)'] + 0.5871 * dataframeoutput['G (15)'] + 0.1140 * dataframeoutput['B (15)']

        # Use a Riemann sum to make a single weighted RGB function
        if RGB_combined:
            # First the averages of the rings need to be computed, instead of the averages of the circles
            # These calculations are specific for the ring set-up and no longer work when different angles are used
            ring_area_list = [531, 1593, 2654, 3717, 4778, 12743, 27077, 66366, 96192, 124398, 157592]
            ring_weight_list = [0.00496552, 0.014728342, 0.024052213, 0.032695798, 0.040386081, 0.099636517, 0.176575706, 0.284684416, 0.195252643, 0.093120638, 0.033902126]
            # pre allocate ring value vectors and reset the total ring weight
            RGB_1_ring = dataframeoutput['Segment ID']/dataframeoutput['Segment ID']
            RGB_1_5_ring = dataframeoutput['Segment ID']/dataframeoutput['Segment ID']
            RGB_2_ring = dataframeoutput['Segment ID']/dataframeoutput['Segment ID']
            RGB_2_5_ring = dataframeoutput['Segment ID']/dataframeoutput['Segment ID']
            RGB_3_5_ring = dataframeoutput['Segment ID']/dataframeoutput['Segment ID']
            RGB_5_ring = dataframeoutput['Segment ID']/dataframeoutput['Segment ID']
            RGB_7_5_ring = dataframeoutput['Segment ID']/dataframeoutput['Segment ID']
            RGB_10_ring = dataframeoutput['Segment ID']/dataframeoutput['Segment ID']
            RGB_12_5_ring = dataframeoutput['Segment ID']/dataframeoutput['Segment ID']
            RGB_15_ring = dataframeoutput['Segment ID']/dataframeoutput['Segment ID']
            RGB_combined_feature = dataframeoutput['Segment ID']/dataframeoutput['Segment ID']
    
            # Compute the RGB average of the rings by the hand of the areas
            # First check if the ring is still in reach, otherwise keep out of the weighted average 
            for k in range(segmentlength*frequency):
                total_ring_weight = ring_weight_list[0]
                if dataframeoutput['RGB 1'][k] != -9999:
                    RGB_1_ring[k] = (dataframeoutput['RGB 1'][k]*(ring_area_list[0]+ring_area_list[1])/ring_area_list[1] - dataframeoutput['RGB 0.5'][k]*ring_area_list[0]/ring_area_list[1])
                    total_ring_weight = total_ring_weight + ring_weight_list[1]
                else:
                    RGB_1_ring[k] = 0
                    
                if dataframeoutput['RGB 1.5'][k] != -9999:
                    RGB_1_5_ring[k] = (dataframeoutput['RGB 1.5'][k]*(ring_area_list[1]+ring_area_list[2])/ring_area_list[2] - dataframeoutput['RGB 1'][k]*ring_area_list[1]/ring_area_list[2])
                    total_ring_weight = total_ring_weight + ring_weight_list[2]
                else:
                    RGB_1_5_ring[k] = 0
                    
                if dataframeoutput['RGB 2'][k] != -9999:
                    RGB_2_ring[k] = (dataframeoutput['RGB 2'][k]*(ring_area_list[2]+ring_area_list[3])/ring_area_list[3] - dataframeoutput['RGB 1.5'][k]*ring_area_list[2]/ring_area_list[3])
                    total_ring_weight = total_ring_weight + ring_weight_list[3]
                else:
                    RGB_2_ring[k] = 0
                    
                if dataframeoutput['RGB 2.5'][k] != -9999:
                    RGB_2_5_ring[k] = (dataframeoutput['RGB 2.5'][k]*(ring_area_list[3]+ring_area_list[4])/ring_area_list[4] - dataframeoutput['RGB 2'][k]*ring_area_list[3]/ring_area_list[4])
                    total_ring_weight = total_ring_weight + ring_weight_list[4]
                else:
                    RGB_2_5_ring[k] = 0

                if dataframeoutput['RGB 3.5'][k] != -9999:
                    RGB_3_5_ring[k] = (dataframeoutput['RGB 3.5'][k]*(ring_area_list[4]+ring_area_list[5])/ring_area_list[5] - dataframeoutput['RGB 2.5'][k]*ring_area_list[4]/ring_area_list[5])
                    total_ring_weight = total_ring_weight + ring_weight_list[5]
                else:
                    RGB_3_5_ring[k] = 0

                if dataframeoutput['RGB 5'][k] != -9999:
                    RGB_5_ring[k] = (dataframeoutput['RGB 5'][k]*(ring_area_list[5]+ring_area_list[6])/ring_area_list[6] - dataframeoutput['RGB 3.5'][k]*ring_area_list[5]/ring_area_list[6])
                    total_ring_weight = total_ring_weight + ring_weight_list[6]
                else:
                    RGB_5_ring[k] = 0

                if dataframeoutput['RGB 7.5'][k] != -9999:
                    RGB_7_5_ring[k] = (dataframeoutput['RGB 7.5'][k]*(ring_area_list[6]+ring_area_list[7])/ring_area_list[7] - dataframeoutput['RGB 5'][k]*ring_area_list[6]/ring_area_list[7])
                    total_ring_weight = total_ring_weight + ring_weight_list[7]
                else:
                    RGB_7_5_ring[k] = 0

                if dataframeoutput['RGB 10'][k] != -9999:
                    RGB_10_ring[k] = (dataframeoutput['RGB 10'][k]*(ring_area_list[7]+ring_area_list[8])/ring_area_list[8] - dataframeoutput['RGB 7.5'][k]*ring_area_list[7]/ring_area_list[8])
                    total_ring_weight = total_ring_weight + ring_weight_list[8]
                else:
                    RGB_10_ring[k] = 0
                    
                if dataframeoutput['RGB 12.5'][k] != -9999:
                    RGB_12_5_ring[k] = (dataframeoutput['RGB 12.5'][k]*(ring_area_list[8]+ring_area_list[9])/ring_area_list[9] - dataframeoutput['RGB 10'][k]*ring_area_list[8]/ring_area_list[9])
                    total_ring_weight = total_ring_weight + ring_weight_list[9]
                else:
                    RGB_12_5_ring[k] = 0

                if dataframeoutput['RGB 15'][k] != -9999:
                    RGB_15_ring[k] = (dataframeoutput['RGB 15'][k]*(ring_area_list[9]+ring_area_list[10])/ring_area_list[10] - dataframeoutput['RGB 12.5'][k]*ring_area_list[9]/ring_area_list[10])
                    total_ring_weight = total_ring_weight + ring_weight_list[10]
                else:
                    RGB_15_ring[k] = 0
                    
                RGB_combined_feature[k] = (dataframeoutput['RGB 0.5'][k]*ring_weight_list[0] + RGB_1_ring[k]*ring_weight_list[1] + RGB_1_5_ring[k]*ring_weight_list[2] + RGB_2_ring[k]*ring_weight_list[3] + RGB_2_5_ring[k]*ring_weight_list[4] + RGB_3_5_ring[k]*ring_weight_list[5] + RGB_5_ring[k]*ring_weight_list[6] + RGB_7_5_ring[k]*ring_weight_list[7] + RGB_10_ring[k]*ring_weight_list[8] + RGB_12_5_ring[k]*ring_weight_list[9] + RGB_15_ring[k]*ring_weight_list[10]) / total_ring_weight
            
            dataframeoutput['RGB Combined'] = RGB_combined_feature
            
        if amountblinks:
            amountblinklist = []
            blinks = 0
            if dataframeoutput['Left Eye_Pupil X'].iloc[0] == 0:
                blinks = blinks + 1
            if dataframeoutput['Left Eye_Pupil X'].iloc[-1] == 0:
                blinks = blinks + 1
                
            blinkcheck = dataframeoutput.drop(dataframeoutput[dataframeoutput['Left Eye_Pupil X']==0].index)
            blinkcheck2 = blinkcheck['Time in Recording']-blinkcheck['Time in Recording'].shift(1)
            blinks = blinks + len(blinkcheck[blinkcheck2 > 0.034])
            amountblinklist.append(blinks)
            
            blinks = 0
            if dataframeoutput['Right Eye_Pupil X'].iloc[0] == 0:
                blinks = blinks + 1
            if dataframeoutput['Right Eye_Pupil X'].iloc[-1] == 0:
                blinks = blinks + 1
                
            blinkcheck = dataframeoutput.drop(dataframeoutput[dataframeoutput['Right Eye_Pupil X']==0].index)
            blinkcheck2 = blinkcheck['Time in Recording']-blinkcheck['Time in Recording'].shift(1)
            blinks = blinks + len(blinkcheck[blinkcheck2 > 0.034])
            amountblinklist.append(blinks)
            
            blinks = 0
            if dataframeoutput['R (0.5)'].iloc[0] == -1000:
                blinks = blinks + 1
            if dataframeoutput['R (0.5)'].iloc[-1] == -1000:
                blinks = blinks + 1
                
            blinkcheck = dataframeoutput.drop(dataframeoutput[dataframeoutput['R (0.5)']==-1000].index)
            blinkcheck2 = blinkcheck['Time in Recording']-blinkcheck['Time in Recording'].shift(1)
            blinks = blinks + len(blinkcheck[blinkcheck2 > 0.034])
            amountblinklist.append(blinks)
            
            dataframeoutput['Amount of Blinks in Segment']=[statistics.median(amountblinklist)]*len(t_vectors[j])

        if PERCLOS_feature:
            PERCLOS_vector = [len(dataframeoutput[dataframeoutput['Left Eye_Pupil X']==0]),len(dataframeoutput[dataframeoutput['Right Eye_Pupil X']==0]),len(dataframeoutput[dataframeoutput['Gaze X']==0])]
            PERCLOS = statistics.median(PERCLOS_vector)/len(dataframeoutput)
            dataframeoutput['PERCLOS']=[PERCLOS]*len(t_vectors[j])

        outputname = file.replace('csv_input\\','')
        outputname = outputname.replace('CsvData.'+csv_input_format,'Segmented.'+csv_output_format)
        
        if j == 0:
            dataframeoutput.to_csv('csv_output/'+outputname, mode = 'w', header = True, index = False)
        else:
            dataframeoutput.to_csv('csv_output/'+outputname, mode = 'a', header = False, index = False)

        # update segement ID
        SegmentID = SegmentID + 1

    # create RGB without blinks from completed dataframe, overwrite
    if RGB_combined:
        dataframeRGB = pd.read_csv('csv_output/'+outputname)
        interpdiam0 = round(dataframeRGB['RGB Combined'].shift(-1)) == -1000
        interpdiam1 = round(dataframeRGB['RGB Combined']) == -1000
        interpdiam2 = round(dataframeRGB['RGB Combined'].shift(1)) == -1000
        interpdiam3 = round(dataframeRGB['RGB Combined'].shift(2)) == -1000
        interpdiam4 = round(dataframeRGB['RGB Combined'].shift(3)) == -1000
        interpdiam5 = round(dataframeRGB['RGB Combined'].shift(4)) == -1000
        interpdiam6 = round(dataframeRGB['RGB Combined'].shift(-1)) == -9999
        interpdiam7 = round(dataframeRGB['RGB Combined']) == -9999
        interpdiam8 = round(dataframeRGB['RGB Combined'].shift(1)) == -9999
        interpdiam9 = round(dataframeRGB['RGB Combined'].shift(2)) == -9999
        interpdiam10 = round(dataframeRGB['RGB Combined'].shift(3)) == -9999
        interpdiam11 = round(dataframeRGB['RGB Combined'].shift(4)) == -9999
        interpdiam = interpdiam0 | interpdiam1 | interpdiam2 | interpdiam3 | interpdiam4 | interpdiam5 | interpdiam6 | interpdiam7 | interpdiam8 | interpdiam9 | interpdiam10 | interpdiam11
        interpdiam = [not t for t in interpdiam]
        dataframeRGB_corrected = dataframeRGB[interpdiam]
        RGB_corrected_func = scipy.interpolate.interp1d(dataframeRGB_corrected['Time in Recording'],dataframeRGB_corrected['RGB Combined'],kind=interpmethod, fill_value='extrapolate')
        RGB_corrected_vector = RGB_corrected_func(dataframeRGB['Time in Recording'])
        insert_location = dataframeRGB.columns.get_loc('RGB Combined') + 1 
        dataframeRGB.insert(insert_location, 'RGB Combined Corrected', RGB_corrected_vector)
        dataframeRGB.to_csv('csv_output/'+outputname, mode = 'w', header = True, index = False)
        for header in allheaders:
            if 'RGB Combined Corrected' not in header:
                insert_location = header.index('RGB Combined') + 1
                header.insert(insert_location,'RGB Combined Corrected')
                
    # If the header is unique, so the set of features is different all previous, add to list of headers
    if list(dataframeRGB.columns) not in allheaders:
        allheaders.append(list(dataframeRGB.columns))

    # update input ID
    i = i + 1  
    
    # If necessary make sure next file is scanned for dataset types
    if gazereset:
        gazedataset = 'scan'
    if eyereset:
        eyedataset = 'scan'
        
# For all unique headers compile a file with the dataframes appended
for i in range(len(allheaders)):
    j = 0
    for file in glob.glob("csv_output/*_Segmented."+csv_output_format):
        dataframeread = pd.read_csv(file)
        if allheaders[i] == list(dataframeread.columns):
            if j == 0:
                dataframeread.to_csv('csv_output/segmentslist%03d.' %i + csv_output_format, mode = 'w', header = True, index = False)
                j = j + 1
            else:
                dataframeread.to_csv('csv_output/segmentslist%03d.' %i + csv_output_format, mode = 'a', header = False, index = False)