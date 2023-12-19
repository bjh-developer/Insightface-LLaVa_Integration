'''
nir max_sim threshold.py takes one image from NIR image folder to be the template image for comparison.
It then runs through all the other images in NIR image folder that does not contain the person in the template image as test images.
As all the test images should not match with the template image, we can take the highest recorded similarity score between the test and template images to be the suitable maximum similarity threshold.

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


mypath = "/data2/Intern/profiling insightface/NIR images/"
template_file = "s1_NIR_00001_001.bmp"
template_matrix = None


print(template_file)
img = cv2.imread(mypath + template_file)
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


onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and 'NIR_00001' not in f]
print(onlyfiles)


count = 0


#Comparing test image to template image
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

    
    
#Plot line graph: the highest point on the line graph is the suitable maximum similarity score threshold to determine if the images matches one another  
import matplotlib.pyplot as plt
import numpy as np


xList = []
yList = []
count = 0
NIR_threshold = 0


for i in test_img_path:
  count += 1
  xList.append(count)
  yList.append(max_sim_list[test_img_path.index(i)])


xpoints = np.array(xList)
ypoints = np.array(yList)


def annot_max(xpoints,ypoints, ax=None):
    global NIR_threshold
    xmax = xpoints[np.argmax(ypoints)]
    ymax = ypoints.max()
    NIR_threshold = ymax
    text= "threshold to set = {:.3f}".format(ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)


annot_max(xpoints,ypoints)
plt.plot(xpoints, ypoints)
plt.ylabel("max_sim")
plt.xlabel("Non-matching photos sample size")
plt.savefig('threshold_for_NIR_image.png')

