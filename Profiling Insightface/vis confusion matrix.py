'''
vis confusion matrix.py takes the first VIS image of everyone's face to be used as template.
Afterwards, it tests all the remaining VIS images against the template
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


mypath = "/data2/Intern/profiling insightface/VIS images/"


onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('001.jpg')]
onlyfiles.sort()


template_matrix = None
count = 0
matching_library = {}


# addind the embed of all the images in onlyfiles into template_matrix
for file in onlyfiles:
    print(file)
    img = cv2.imread(mypath + file)
    faces = app.get(img)
  

    #Checks how many faces are there in the image
    #If only 1 face, proceed with getting template matrix
    #If > 1 face, crop out the face closest to the centre of the image and use that face as template
    if len(faces) == 1:
        face = faces[0]
        embed = face.normed_embedding
        embed = embed.reshape(1, -1)
        template_matrix = embed
      
    elif len(faces) == 0:
        pass
    
    else:
        #Gets the coordinates of the centre of the image
        h, w, c = img.shape
        hCentre = h/2
        wCentre = w/2
        correct_face = None
        best_distance = None
        for face in faces:
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
    
      embed = correct_face.normed_embedding
      embed = embed.reshape(1, -1)
  
  
max_sim_list = []
template_match_list = []
test_img_path = []


onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and not f.endswith('001.jpg')]
onlyfiles.sort()


count = 0


# comparing test image to template image
for file in onlyfiles:
    count += 1
    test_img_path.append(file)
    img = cv2.imread(mypath + file)
    faces = app.get(img)
  
  
    if len(faces) == 1:
        face = faces[0]
        embed = face.normed_embedding
        embed = embed.reshape(1, -1)
  
  
    elif len(faces) == 0:
        pass
  
  
    else:
        h, w, c = img.shape
        hCentre = h/2
        wCentre = w/2
    
        correct_face = None
        best_distance = None
        for face in faces:
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
                    
        embed = correct_face.normed_embedding
        embed = embed.reshape(1, -1)
  
  
    sim = np.matmul(template_matrix, embed.transpose())
    max_sim_index = np.argmax(sim)
    max_sim = sim[max_sim_index]
    template_match_list.append(max_sim_index)
    max_sim_list.append(max_sim)
  
  
    print(test_img_path[-1])
    print(template_match_list[-1])
    print(max_sim_list[-1])
    print(count)
    print()
    print('---------------------------------')
    print()



count = -1
actual = np.array([])
predicted = np.array([])


#generates an array of actual and predicted values.
#actual array values of 1 and 0, 1 represents the test and template image should match in reality, 0 represents the opposite
#predicted array values of 1 and 0, 1 represents the test and template image did match from insightface, 0 represents the opposite
for i in test_img_path:
  count += 1

  if i[:12] in matching_library[template_match_list[count]]:
    actual = np.append(actual, [1])
  else:
    actual = np.append(actual, [0])

  if max_sim_list[test_img_path.index(i)] >= 0.341:
    predicted = np.append(predicted, [1])
  else:
    predicted = np.append(predicted, [0])


print(actual)
print()
print(predicted)



#generates the confusion matrix
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

print(actual, predicted)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.savefig('confusion_matrix_for_VIS.png')