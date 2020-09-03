# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:12:40 2019

This script takes the processed data and calculates the features. The calculated
features are then passed to a Random Forest classifier. Several evaluation metrics
are calculated, many optional. Take settings and proband exclusion into account.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
from tsfresh.feature_extraction import feature_calculators

############################# SETTINGS ##############################
# The features you wish to take from the datasheets with a calculation mean and variance
DesiredFeatures_mean_and_var = ['Right Eye_Pupil Diameter Corrected','Left Eye_Pupil Diameter Corrected','RGB Combined Corrected']
# Features of which a variance doesn't make sense, like PERCLOS
DesiredFeatures_no_var = ['PERCLOS','Gaze X Corrected','Gaze Y Corrected']

# Join the different circumstances into 1 class
join_classes = False

# If true, single train/test split (80/20). If false, 10-fold crossvalidation
quick_run = False

# If true, compute and plot learning curve. Only meaningfull when join_classes = True.
learningcurve = True

# If true, compute confusion matrices
Confusion = True

# If true, exclude outliers with Isolation Forest, not recommended
IsoForest = False

# Set random state to compare small changes fairly
rstate = 1 
#####################################################################
DesiredFeatures = DesiredFeatures_mean_and_var + DesiredFeatures_no_var

# Read in the dataframes of all the classes
class0 = pd.read_csv('Study Silver Disk no smoothing/BaselineE/segmentslist000.csv')
class1 = pd.read_csv('Study Silver Disk no smoothing/BaselineS/segmentslist000.csv')
class2 = pd.read_csv('Study 1 (Test Niko) no smoothing/Baseline/segmentslist000.csv')

class3 = pd.read_csv('Study Silver Disk no smoothing/1 back E/segmentslist000.csv')
class4 = pd.read_csv('Study Silver Disk no smoothing/1 back S/segmentslist000.csv')
class5 = pd.read_csv('Study 1 (Test Niko) no smoothing/1 back/segmentslist000.csv')

class6 = pd.read_csv('Study Silver Disk no smoothing/2 back E/segmentslist000.csv')
class7 = pd.read_csv('Study Silver Disk no smoothing/2 back S/segmentslist000.csv')
class8 = pd.read_csv('Study 1 (Test Niko) no smoothing/2 back/segmentslist000.csv')

# Exclude the following Probands from analysis per class
class0exclude = [15] # Baseline E (easy road)
class1exclude = [29] # Baseline S (hard road)
class3exclude = [38,21] # 1 back E (easy road)
class4exclude = [29] # 1 back S (hard road)
class6exclude = [34] # 2 back E (easy road)
class7exclude = [31] # 2 back S (hard road)

# For study 1 please deliver as strings if below 10
class2exclude = ['36','06','08','18','26','29','28','33','38'] # Baseline (Study 1)
class5exclude = ['36','06','08'] # 1 back (Study 1)
class8exclude = ['36','08','02'] # 2 back (Study 1)

for z in class0exclude:
    class0 = class0[[not j for j in class0['Filename Input'].str.contains('Proband'+str(z))]]
for z in class1exclude:
    class1 = class1[[not j for j in class1['Filename Input'].str.contains('Proband'+str(z))]]
for z in class3exclude:
    class3 = class3[[not j for j in class3['Filename Input'].str.contains('Proband'+str(z))]]
for z in class4exclude:
    class4 = class4[[not j for j in class4['Filename Input'].str.contains('Proband'+str(z))]]
for z in class6exclude:
    class6 = class6[[not j for j in class6['Filename Input'].str.contains('Proband'+str(z))]]
for z in class7exclude:
    class7 = class7[[not j for j in class7['Filename Input'].str.contains('Proband'+str(z))]]

for z in class2exclude:
    class2 = class2[[not j for j in class2['Filename Input'].str.contains('_'+str(z)+'_')]]
for z in class5exclude:
    class5 = class5[[not j for j in class5['Filename Input'].str.contains('_'+str(z)+'_')]]
for z in class8exclude:
    class8 = class8[[not j for j in class8['Filename Input'].str.contains('_'+str(z)+'_')]]

# Make one nested list with all data so it can be looped over
nested_datalist = []
nested_datalist.append(class0[DesiredFeatures])
nested_datalist.append(class1[DesiredFeatures])
nested_datalist.append(class2[DesiredFeatures])
nested_datalist.append(class3[DesiredFeatures])
nested_datalist.append(class4[DesiredFeatures])
nested_datalist.append(class5[DesiredFeatures])
nested_datalist.append(class6[DesiredFeatures])
nested_datalist.append(class7[DesiredFeatures])
nested_datalist.append(class8[DesiredFeatures])

# Calculate the amounts of datapoints in one segment, should be equal for all
segmentlength = len(class0[class0['Segment ID']==0])

# Calculate the amount of segments
segments = int((sum(len(x) for x in nested_datalist))/segmentlength)

# Create empty list that will contain the columns for the RF dataframe
RFcolumns = []

#Compute the columns for the RF dataframe
for column in nested_datalist[0].columns:
    if column in DesiredFeatures_mean_and_var:
        RFcolumns.append('median '+column)
        RFcolumns.append('relative change '+column)
    elif column in DesiredFeatures_no_var:
        RFcolumns.append('median '+column)
    else:
        RFcolumns.append('median '+column)
RFcolumns.append('Class')
    
# Create an empty dataframe that will contain the data as the RF algorithm will receive it
RFmatrix = pd.DataFrame(index = range(segments),columns = RFcolumns)

# Create a counter that continues while we loop over classes
seg_counter = 0
# Loop over classes
for j in range(len(nested_datalist)):
    # Loop over segments per class
    for i in range(int(len(nested_datalist[j])/segmentlength)):
        # Get the data from the individual segment
        segment_matrix = nested_datalist[j][i*segmentlength:(i+1)*segmentlength]
        feature_counter = 0
        # Loop over the data and compute mean and relative changes, add the Class in a seperate column
        for column in segment_matrix.columns:
            feature_vector = segment_matrix[column]
            if column in DesiredFeatures_mean_and_var:
                RFmatrix.iloc[seg_counter,feature_counter] = np.median(feature_vector)
                feature_counter = feature_counter + 1
                RFmatrix.iloc[seg_counter,feature_counter] = feature_calculators.absolute_sum_of_changes(feature_vector)/np.mean(feature_vector)
                feature_counter = feature_counter + 1
            else:
                RFmatrix.iloc[seg_counter,feature_counter] = np.median(feature_vector)
                feature_counter = feature_counter + 1
        # make classes 0 and 1 become 0, 2 and 3 become 1, etc.
        if join_classes:
            RFmatrix.iloc[seg_counter]['Class'] = np.floor(j/3)
        else:
            RFmatrix.iloc[seg_counter]['Class'] = j
        seg_counter = seg_counter + 1

RFmatrix = RFmatrix[RFmatrix['median Right Eye_Pupil Diameter Corrected']<90]
RFmatrix = RFmatrix[RFmatrix['median Left Eye_Pupil Diameter Corrected']<90]
RFmatrix = RFmatrix[RFmatrix['median Right Eye_Pupil Diameter Corrected']>20]
RFmatrix = RFmatrix[RFmatrix['median Left Eye_Pupil Diameter Corrected']>20]
RFmatrix = RFmatrix[RFmatrix['median RGB Combined Corrected']<255]
RFmatrix = RFmatrix[RFmatrix['median RGB Combined Corrected']>0]
RFmatrix = RFmatrix[RFmatrix['median PERCLOS']<0.25]

RFmatrix = RFmatrix.dropna()       

# Exclude outliers with Isolation Forest
if IsoForest:
    ifc = IsolationForest(random_state = rstate, contamination = 0.05).fit(RFmatrix[['median PERCLOS','Class']])
    RFmatrixIso = ifc.predict(RFmatrix[['median PERCLOS','Class']])
    RFmatrix = RFmatrix[RFmatrixIso==1]


# Split in features and labels 
RFmatrix_X = RFmatrix.drop(columns=['Class'],axis=1).astype(float)
RFmatrix_y = RFmatrix['Class']
RFmatrix_y=RFmatrix_y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(RFmatrix_X, RFmatrix_y, test_size=0.2,random_state = rstate)

clf = RandomForestClassifier(n_estimators=1000, min_samples_split=2,min_samples_leaf=1,max_features='sqrt',max_depth=None, 
                             bootstrap = False, n_jobs = -1, random_state=rstate)
clf.fit(X_train.astype(float),y_train.astype(float))


X_shuf, Y_shuf = shuffle(RFmatrix_X, RFmatrix_y,random_state = rstate)
if learningcurve:
    trainsizes = int(np.floor((0.9*len(RFmatrix))/100)*100)
    trainsizes = np.linspace(100,trainsizes,int(trainsizes/100),dtype=int).tolist()
    train_sizes, train_scores, validation_scores = learning_curve(estimator = RandomForestClassifier(n_estimators=1000, 
                                                                                                     min_samples_split=2,
                                                                                                     min_samples_leaf=1,
                                                                                                     max_features='sqrt',
                                                                                                     max_depth=None, 
                                                                                                     bootstrap = False, 
                                                                                                     n_jobs = -1, 
                                                                                                     random_state=rstate),
                                                                  X = X_shuf, y = Y_shuf, train_sizes = trainsizes, cv = 10, n_jobs = -1)
    plt.figure()
    plt.plot(trainsizes,np.mean(validation_scores,1))
    plt.title('Learning curve Random Forest Cognitive Load')
    plt.ylim(0,1)
    plt.ylabel('Accuracy in 10-fold crossvalidation')
    plt.xlabel('Training samples')
    plt.grid()
    
y_pred=clf.predict(X_test)
print('Confusion matrix:\n',metrics.confusion_matrix(y_test,y_pred,normalize='true'))

if quick_run:
    print("\nAccuracy:",metrics.accuracy_score(y_test, y_pred))
else:
    scores = cross_val_score(RandomForestClassifier(n_estimators=1000, min_samples_split=2,
                                                    min_samples_leaf=1,max_features='sqrt',
                                                    max_depth=None, bootstrap = False, 
                                                    n_jobs = -1, random_state=rstate),
                             X_shuf, Y_shuf, cv=10, n_jobs = -1)
    print("\nAccuracy:",np.mean(scores))

plt.figure()
plt.bar(range(len(clf.feature_importances_)),clf.feature_importances_)
plt.title('Feature importance, sums to 1')
labels = ['Median right pupil diameter', 'RCPD right',
          'Median left pupil diameter', 'RCPD left',
          'median Light Intensity','RCLI',
          'PERCLOS',
          'Median gaze x coordinate','Median gaze y coordinate']
plt.xticks(range(len(RFmatrix_X.columns)),labels)
plt.xticks(fontsize=8,rotation=90)

if Confusion:
    # Plot non-normalized confusion matrix
    plt.figure()
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                     #display_labels=['Baseline','1-back','2-back'],
                                     display_labels=['Baseline Easy Road','Baseline Hard Road','Baseline Study 1',
                                                     '1-back Easy Road','1-back Hard Road','1-back Study 1','2-back Easy Road',
                                                     '2-back Hard Road','2-back Study 1','3-back Study 1'],
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        plt.xticks(rotation=90)
        disp.ax_.set_title(title)
    plt.show()