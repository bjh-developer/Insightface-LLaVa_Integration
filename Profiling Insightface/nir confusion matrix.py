'''
nir confusion matrix.py takes the first NIR image of everyone's face to be used as template.
Afterwards, it tests all the remaining NIR images against the template
and churns out a confusion matrix to visualise the confusion by Insightface.

'''


import argparse
import cv2
import sys
import numpy as np
import insightface
import glob
from matplotlib import pyplot as plt
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import math
from numpy.linalg import norm
from os import listdir
from os.path import isfile, join


app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))


NIR_mypath = "/data2/Intern/profiling insightface/NIR images/"


NIR_onlyfiles = [f for f in listdir(NIR_mypath) if isfile(join(NIR_mypath, f)) and f.endswith('001.bmp')]
NIR_onlyfiles.sort()


NIR_template_matrix = None
count = 0
NIR_matching_library = {}


# addind the embed of all the images in onlyfiles into template_matrix
for NIR_file in NIR_onlyfiles:
    print(NIR_file)
    NIR_img = cv2.imread(NIR_mypath + NIR_file)
    NIR_faces = app.get(NIR_img)
  
    # Should only have 1 face per image, to do some filtering if not true
    if len(NIR_faces) == 1:
        count += 1
        NIR_face = NIR_faces[0]
        NIR_embed = NIR_face.normed_embedding
    
        # Convert to 1 x feature_length matrix
        NIR_embed = NIR_embed.reshape(1, -1)
    
        if NIR_template_matrix is None:
            NIR_template_matrix = NIR_embed
        else:
            NIR_template_matrix = np.append(NIR_template_matrix, NIR_embed, axis=0)
        # adds the key-value pair into the dictionary
        NIR_matching_library[count-1] = NIR_file
  
    elif len(NIR_faces) == 0:
        pass
    
    else:
        #Gets the coordinates of the centre of the image
        h, w, c = NIR_img.shape
        hCentre = h/2
        wCentre = w/2
        correct_face = None
        best_distance = None
        for face in NIR_faces:
            box = face.bbox
            xBoxCentre = ((box[2]-box[0])/2)+box[0]
            yBoxCentre = ((box[3]-box[1])/2)+box[1]
            #Distance of box centre to photo centre
            distance = math.sqrt((xBoxCentre-wCentre)**2+(yBoxCentre-hCentre)**2)
            if best_distance == None:
                best_distance = distance
                correct_face = face
            elif distance < best_distance:
                best_distance = distance
                correct_face = face
    
        NIR_embed = correct_face.normed_embedding
        NIR_embed = NIR_embed.reshape(1, -1)
        
        if NIR_template_matrix is None:
            NIR_template_matrix = NIR_embed
        else:
            NIR_template_matrix = np.append(NIR_template_matrix, NIR_embed, axis=0)
        # adds the key-value pair into the dictionary
        NIR_matching_library[count-1] = NIR_file


NIR_max_sim_list = []
NIR_template_match_list = []
NIR_test_img_path = []


NIR_onlyfiles = [f for f in listdir(NIR_mypath) if isfile(join(NIR_mypath, f)) and not f.endswith('001.bmp')]
NIR_onlyfiles.sort()


count = 0


# comparing test image to template image
for NIR_file in NIR_onlyfiles:
  
    NIR_img = cv2.imread(NIR_mypath + NIR_file)
    NIR_faces = app.get(NIR_img)
  
    # Should only have 1 face per image, to do some filtering if not true
    if len(NIR_faces) == 1:
        count += 1
        NIR_test_img_path.append(NIR_file)
        NIR_face = NIR_faces[0]
        NIR_embed = NIR_face.normed_embedding

    elif len(NIR_faces) == 0:
            pass
      
    else:
        h, w, c = NIR_img.shape
        hCentre = h/2
        wCentre = w/2
    
        correct_face = None
        best_distance = None
        for face in NIR_faces:
            box = face.bbox
            xBoxCentre = ((box[2]-box[0])/2)+box[0]
            yBoxCentre = ((box[3]-box[1])/2)+box[1]
            distance = math.sqrt((xBoxCentre-wCentre)**2+(yBoxCentre-hCentre)**2)
            if best_distance == None:
                best_distance = distance
                correct_face = face
            else:
                if distance < best_distance:
                    best_distance = distance
                    correct_face = face
                    
        NIR_embed = correct_face.normed_embedding
        embed = NIR_embed.reshape(1, -1)
  
  
    sim = np.matmul(NIR_template_matrix, NIR_embed.transpose())
    max_sim_index = np.argmax(sim)
    max_sim = sim[max_sim_index]
    NIR_template_match_list.append(max_sim_index)
    NIR_max_sim_list.append(max_sim)
  
  
    print(NIR_test_img_path[-1])
    print(NIR_template_match_list[-1])
    print(NIR_max_sim_list[-1])
    print(count)
    print()
    print('---------------------------------')
    print()



count = -1
NIR_actual = np.array([])
NIR_predicted = np.array([])


#generates an array of actual and predicted values.
#actual array values of 1 and 0, 1 represents the test and template image should match in reality, 0 represents the opposite
#predicted array values of 1 and 0, 1 represents the test and template image did match from insightface, 0 represents the opposite
for i in NIR_test_img_path:
    count += 1
  
    if i[:12] in NIR_matching_library[NIR_template_match_list[count]]:
        NIR_actual = np.append(NIR_actual, [1])
    else:
        NIR_actual = np.append(NIR_actual, [0])
  
    if NIR_max_sim_list[NIR_test_img_path.index(i)] >= 0.443:
        NIR_predicted = np.append(NIR_predicted, [1])
    else:
        NIR_predicted = np.append(NIR_predicted, [0])

print(NIR_actual)
print()
print(NIR_predicted)



#generates the confusion matrix
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

print(NIR_actual)
print(NIR_predicted)

confusion_matrix = metrics.confusion_matrix(NIR_actual, NIR_predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.savefig('confusion_matrix_for_NIR.png')

