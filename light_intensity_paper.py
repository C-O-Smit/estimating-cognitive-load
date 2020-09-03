# -*- coding: utf-8 -*-
"""
The goal of the script is to extract the light intensity in certain areas from
video frames. 

For practical reasons this script uses whole videos instead of segments. The
segmentation is applied to the lists produced by this script instead of on the 
input videos.

The script assumes that the frequency of the markers is double the frequency 
of the front facing camera. This is the case for the Dikablis Glasses 3 and the
Dikablis Professional.
"""

import os
import glob
import numpy as np
import cv2
import csv
import pandas as pd
import sys

######################### SETTINGS ############################
# Set diameter range to obtain pixel values, in degrees measured from line of sight
# If you change this please be advised that you will need to change the writing to the .csv as well
diameterrange = [0.5,1,1.5,2,2.5,3.5,5,7.5,10,12.5,15]

# The vertical view angle in degrees of the entire image measured from the line of sight
# For the Dikablis Glasses 3 this is 30 degrees, for the Dikablis Professional this is 21 degrees
totalviewangle = 30 

# Format of the videos you want to examine
vidformat = "mp4"

# Format of the csv input data (csv or txt)
csv_input_format = "txt"

# Format of the csv output data (csv or txt)
csv_output_format = "csv"

# Gaze Data set name ('scan' will take the first available option, or set to e.g. 'Original', 'Processed')
gazedataset = 'scan'
###############################################################

# if the 'csv_output' directory is not there, create it
if not os.path.isdir("csv_output"):
    os.mkdir("csv_output")

# finding all video files of the set format in folder and adding them to a csv list
path = os.path.dirname(__file__) # get current path
os.chdir(path) # change path to folder of .py file
with open('csv_output/VideoList.'+csv_output_format, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['Filename','Index'])
csvFile.close()

# adding all mp4 files in folder to VideoList.csv and indexing them
i = 0
for file in glob.glob("videos/*."+vidformat):
    row = [file,str(i)]
    with open('csv_output/VideoList.'+csv_output_format, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
    i = i + 1    

# create mismatch lists for diagnostic purposes
mismatchlist = []
mismatchtypelist = []
matchingstrategylist = []

# for all videos in VideoList.csv, get the frames and place them in directory train_1    
with open('csv_output/VideoList.'+csv_output_format, "rt") as VideoCSV:
    reader = csv.reader(VideoCSV)
    
    for row in reader:
        vid = str(row[0])
        
        # skip the header
        if not vid == 'Filename':
            
            if "Dikablis Glasses 3" in vid:
                device = "Dikablis Glasses 3"
            elif "Dikablis Professional" in vid:
                device = "Dikablis Professional"
            else:
                print('Device not recognised. Perhaps filenames are not in the correct format.')
                break
            
            video = cv2.VideoCapture(vid)
            fps = round(video.get(cv2.CAP_PROP_FPS))
            videonr = int(row[1])
            vidpath = path+'/'+vid
            
            # Get the correct path for the .csv or .txt with the gaze markers
            temppath = path+'/videos'
            vidhandler = vidpath.replace(temppath,'')
            vidhandler = vidhandler.replace('_'+device+'_Scene Cam','')
            vidhandler = vidhandler.replace('.'+vidformat,'')
            csvhandler = vidhandler + '_CsvData.' + csv_input_format
            csvpath = path+'/csv_input'+csvhandler
            outputhandler = vidhandler.replace('\\','')
            
            with open('csv_output/'+ outputhandler +'_light_intensity.' + csv_output_format, 'w', newline='') as csvFile:
                writer = csv.writer(csvFile)
            
                row = ['Path video','Filename video','Video ID','Frame','Time','Gaze Marker X','Gaze Marker Y','R (RGB)','G (RGB)','B (RGB)','R (0.5)','G (0.5)','B (0.5)','R (1)','G (1)','B (1)','R (1.5)','G (1.5)','B (1.5)','R (2)','G (2)','B (2)','R (2.5)','G (2.5)','B (2.5)','R (3.5)','G (3.5)','B (3.5)','R (5)','G (5)','B (5)','R (7.5)','G (7.5)','B (7.5)','R (10)','G (10)','B (10)','R (12.5)','G (12.5)','B (12.5)','R (15)','G (15)','B (15)']
                writer.writerow(row)
            csvFile.close()
            
            with open('csv_output/'+ outputhandler +'_light_intensity.' + csv_output_format, 'a', newline='') as csvFile:
                writer = csv.writer(csvFile)
                
                framesleft,image = video.read() # check if there are frames to extract
                resolution = image.shape
                
                # set the frame counter to 0 
                framenr = 0
                
                # Create lists for radii and masks
                radiuslist = []
                circle_mask_list = []
                
                # Get dataframe from csv
                if csv_input_format == "txt":
                    dataframe = pd.read_csv(csvpath, sep = '\t')
                elif csv_input_format == "csv":
                    dataframe = pd.read_csv(csvpath)
                else:
                    print('Please choose a valid format for the csv data')
                    sys.exit()
                    
                if gazedataset == 'scan':
                    columnname_gaze = [col for col in dataframe.columns if '_Gaze_Gaze X' in col]
                    gazedataset = columnname_gaze[0].replace(device+'_Field Data_Scene Cam_','')
                    gazedataset = gazedataset.replace('_Gaze_Gaze X','')
                    gazereset = True
                else:
                    gazereset = False

                # Drop NaN values
                dataframe2 = dataframe[np.isfinite(dataframe[device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze X'])]
                # Get the time to previous and the next data point
                time_to_last = dataframe2['UTC']-dataframe2['UTC'].shift(1)
                time_to_next = dataframe2['UTC'].shift(-1)-dataframe2['UTC']
                # Calculate the time gap that would be there if the point was omitted
                gap_if_point_omitted = time_to_last + time_to_next
                
                # See if time to previous or next data point is lower than 10 ms (should not be possible)
                low_time_to_last = time_to_last < 10
                low_time_to_next = time_to_next < 10
                # Check if time between surrounding points is lower than 25 ms (should not be possible)
                low_gap_if_point_omitted = gap_if_point_omitted < 25
                # See if zero
                checkcolumn = dataframe2[device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze X']
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
                dataframe_last = dataframe2[real_values_last]
                dataframe_next = dataframe2[real_values_next]
                dataframe_omitted = dataframe2[real_values_omitted]
                
                # Count number of frames in video
                length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))-1 # Last frame is unreadable and excluded from analysis
                # Because we omit the last frame, we should omit the last two markers
                dataframe_last.drop(dataframe_last.tail(2).index,inplace=True)
                dataframe_next.drop(dataframe_next.tail(2).index,inplace=True)
                dataframe_omitted.drop(dataframe_omitted.tail(2).index,inplace=True)
                
                markerlength_last = len(dataframe_last[device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze X'])
                markerlength_next = len(dataframe_next[device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze X'])
                markerlength_omitted = len(dataframe_omitted[device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze X'])
                
                mismatch_last = abs(markerlength_last - 2*length)
                mismatch_next = abs(markerlength_next - 2*length)
                mismatch_omitted = abs(markerlength_omitted - 2*length)
                
                # Find the best strategy
                mismatch = np.min([mismatch_last,mismatch_next,mismatch_omitted])
                
                if mismatch == mismatch_last:
                    dataframe3 = dataframe_last
                    matchingstrategy = 'last'
                elif mismatch == mismatch_next:
                    dataframe3 = dataframe_next
                    matchingstrategy = 'next'
                elif mismatch == mismatch_omitted:
                    dataframe3 = dataframe_omitted
                    matchingstrategy = 'omitted'

                # take the relevant collumns
                gazex = dataframe3[device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze X']
                gazey = dataframe3[device+'_Field Data_Scene Cam_'+gazedataset+'_Gaze_Gaze Y']
                timelist = dataframe3['rec_time']
                                                
                # Reset the indices and convert to data-type list
                gazex = list(gazex.reset_index(drop=True))
                gazey = list(gazey.reset_index(drop=True))
                timelist = list(timelist.reset_index(drop=True))
                
                # Calculate the mismatch between frames in the video and amount of remaining gaze markers, print warning if > 10
                mismatch = abs(len(gazex) - 2*length)
                # mismatch of 2 frames is possible due to the different frame rates, more than that is concerning
                if  mismatch > 2:
                    print('\nWARNING: The mismatch between frames in the video and gaze markers in the cleaned list is %i frames, please check if the data is valid' %(mismatch))
                mismatchlist.append(mismatch)
                
                # If there are too many markers, drop the surplus at equal intervals
                if len(gazex) > 2*length:
                    mismatchtype = 'marker surplus'
                    dropnr = len(gazex) - 2*length
                    droprange = range(1,dropnr+1)
                    for i in droprange:
                        gazex = pd.DataFrame(gazex)
                        gazey = pd.DataFrame(gazey)
                        timelist = pd.DataFrame(timelist)
                        gazex = gazex.drop(round((i/(dropnr+1))*len(gazex)))
                        gazey = gazey.drop(round((i/(dropnr+1))*len(gazex)))
                        timelist = timelist.drop(round((i/(dropnr+1))*len(gazex)))
                        gazex = gazex.values.tolist()
                        gazey = gazey.values.tolist()
                        timelist = timelist.values.tolist()
                        
                # If there are too few markers, at equal intervals insert an extra marker by copying previous marker    
                elif len(gazex) < 2*length:
                    mismatchtype = 'marker shortage'
                    dropnr = 2*length - len(gazex)
                    droprange = range(1,dropnr+1)
                    for i in droprange:
                        index = round((i/(dropnr+1))*len(gazex)) 
                        gazex.insert(index,gazex[index])
                        gazey.insert(index,gazey[index])
                        timelist.insert(index,timelist[index])
                        
                elif len(gazex) == 2*length:
                    mismatchtype = 'perfect match'
                
                # Add types of mismatch and solving strategies to diagnostic lists
                mismatchtypelist.append(mismatchtype)
                matchingstrategylist.append(matchingstrategy)
                
                # Create reusable masks for every diameter
                for angle in diameterrange:
                    
                    radius = round(np.tan(angle/180*np.pi)*((resolution[0]/2) / np.tan(totalviewangle/180*np.pi)))
                    radiuslist.append(radius)
                                                                                
                    # Create square the size of the circle
                    x = np.arange(0, 2*radius)
                    y = np.arange(0, 2*radius)
                    
                    # Make a circle shaped mask 
                    circle_mask  = (x[None,:]-radius)**2 + (y[:,None]-radius)**2 <= radius**2
                    circle_mask_list.append(circle_mask)                        
                    
                # While frames are left, get the RGB values
                while framesleft:
                                                                            
                    # Get the mean values of the entire frame
                    b = np.mean(image[:,:,0])
                    g = np.mean(image[:,:,1])
                    r = np.mean(image[:,:,2])
                                                
                    # Take the gaze markers from the cleaned .csv list
                    xMarker = gazex[2*framenr] 
                    if type(xMarker) == list:
                        xMarker = xMarker[0]
                    yMarker = gazey[2*framenr] 
                    if type(yMarker) == list:
                        yMarker = yMarker[0]
                    
                    # Get the marker time
                    markertime = str(timelist[2*framenr])
                    markertime = markertime.replace('[','')
                    markertime = markertime.replace('\'','')
                    markertime = markertime.replace(']','')
                    
                    # Convert to float
                    hours, minutes, seconds = [float(x) for x in markertime.split(':')]
                    t_marker = 3600 * hours + 60 * minutes + seconds
                    
                    rdiam = []
                    gdiam = []
                    bdiam = []
                    
                    # for every diameter get the RGB values
                    j = 0
                    for angle in diameterrange:
                        radius = radiuslist[j]
                            
                        # If the eyes are closed put a large negative value
                        if xMarker == 0 and yMarker == 0:
                            rdiam.append(-1000)
                            gdiam.append(-1000)                                    
                            bdiam.append(-1000)
                            
                        # If the diameter edge falls off the screen put another large negative value
                        elif xMarker - radius < 0 or xMarker + radius > resolution[1] or yMarker - radius < 0 or yMarker + radius > resolution[0]:
                            rdiam.append(-9999)
                            gdiam.append(-9999)                                    
                            bdiam.append(-9999)

                        else:
                            # Get the values from the square enclosing the circle
                            square = image[int(yMarker-radius):int(yMarker+radius),int(xMarker-radius):int(xMarker+radius)]
                            # Get the right diameter mask from the list                                                                        
                            circle_mask = circle_mask_list[j]
                            # Put the mask over the square to create the feature vector
                            circle = square[circle_mask,:]
                            
                            # Append the average values to the lists
                            rdiam.append(np.mean(circle[:,2]))
                            gdiam.append(np.mean(circle[:,1]))
                            bdiam.append(np.mean(circle[:,0]))

                        j = j + 1
                        
                    # write the filename, video and frame to FrameList.csv
                    row = [vidpath,vid,videonr,framenr,t_marker,xMarker,yMarker,r,g,b,rdiam[0],gdiam[0],bdiam[0],rdiam[1],gdiam[1],bdiam[1],rdiam[2],gdiam[2],bdiam[2],rdiam[3],gdiam[3],bdiam[3],rdiam[4],gdiam[4],bdiam[4],rdiam[5],gdiam[5],bdiam[5],rdiam[6],gdiam[6],bdiam[6],rdiam[7],gdiam[7],bdiam[7],rdiam[8],gdiam[8],bdiam[8],rdiam[9],gdiam[9],bdiam[9],rdiam[10],gdiam[10],bdiam[10]]
                    writer.writerow(row)  
                    
                    # For the same frame repeat the process, but with the new marker                    
                    xMarker = gazex[2*framenr+1]
                    if type(xMarker) == list:
                        xMarker = xMarker[0]
                    yMarker = gazey[2*framenr+1] 
                    if type(yMarker) == list:
                        yMarker = yMarker[0]

                    # Get the marker time
                    markertime = str(timelist[2*framenr+1])
                    markertime = markertime.replace('[','')
                    markertime = markertime.replace('\'','')
                    markertime = markertime.replace(']','')

                    # Convert to float
                    hours, minutes, seconds = [float(x) for x in markertime.split(':')]
                    t_marker = 3600 * hours + 60 * minutes + seconds

                    rdiam = []
                    gdiam = []
                    bdiam = []
                    
                    # for every diameter get the RGB values
                    j = 0
                    for angle in diameterrange:
                        radius = radiuslist[j]
                            
                        # If the eyes are closed put a large negative value
                        if xMarker == 0 and yMarker == 0:
                            rdiam.append(-1000)
                            gdiam.append(-1000)                                    
                            bdiam.append(-1000)
                            
                        # If the diameter edge falls off the screen put another large negative value
                        elif xMarker - radius < 0 or xMarker + radius > resolution[1] or yMarker - radius < 0 or yMarker + radius > resolution[0]:
                            rdiam.append(-9999)
                            gdiam.append(-9999)                                    
                            bdiam.append(-9999)

                        else:
                            # Get the values from the square enclosing the circle
                            square = image[int(yMarker-radius):int(yMarker+radius),int(xMarker-radius):int(xMarker+radius)]
                            # Get the right diameter mask from the list                                                                        
                            circle_mask = circle_mask_list[j]
                            # Put the mask over the square to create the feature vector
                            circle = square[circle_mask,:]
                            
                            # Append the average values to the lists
                            rdiam.append(np.mean(circle[:,2]))
                            gdiam.append(np.mean(circle[:,1]))
                            bdiam.append(np.mean(circle[:,0]))

                        j = j + 1
                        
                    # write the filename, video and frame to FrameList.csv
                    row = [vidpath,vid,videonr,framenr,t_marker,xMarker,yMarker,r,g,b,rdiam[0],gdiam[0],bdiam[0],rdiam[1],gdiam[1],bdiam[1],rdiam[2],gdiam[2],bdiam[2],rdiam[3],gdiam[3],bdiam[3],rdiam[4],gdiam[4],bdiam[4],rdiam[5],gdiam[5],bdiam[5],rdiam[6],gdiam[6],bdiam[6],rdiam[7],gdiam[7],bdiam[7],rdiam[8],gdiam[8],bdiam[8],rdiam[9],gdiam[9],bdiam[9],rdiam[10],gdiam[10],bdiam[10]]
                    writer.writerow(row)  
                    
                    framesleft,image = video.read() # read the next frame
                    framenr = framenr + 1 # increase frame count by 1 
                    sys.stdout.write('\rProcessing frame %i out of %i from video %i          ' %(framenr,length,videonr))
                    sys.stdout.flush()
                if gazereset:
                    gazedataset = 'scan'            
            csvFile.close()
VideoCSV.close()