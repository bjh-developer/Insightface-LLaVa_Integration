import argparse
import cv2
import sys
import numpy as np
import insightface
from matplotlib import pyplot as plt
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import math
from numpy.linalg import norm
from os import listdir
from os.path import isfile, join


#initialise insightface's FaceAnalysis model
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

count = 0

def insightface_temp(template, matching_library, template_matrix):

    '''
    insightface_temp function takes in a template image and generates an insightface template in template_matrix
    for insightface_test to use it.
    
    '''
    
    
    global count
    mypath = "/data2/Intern/Insightface-LLaVa/Scraped_images/"
    count += 1
    img = cv2.imread(mypath + template)
    faces = app.get(img)
    
    
    if len(faces) == 1:
        face = faces[0]
        embed = face.normed_embedding
        embed = embed.reshape(1, -1)
        
        if template_matrix is None:
            template_matrix = embed
            
        else:
            template_matrix = np.append(template_matrix, embed, axis=0)
            
            
    else:
        #crops and returns image of the person's face nearest to the centre of the original image
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
        
        if template_matrix is None:
            template_matrix = embed
        else:
            template_matrix = np.append(template_matrix, embed, axis=0)
            
            
    matching_library[count-1] = template
        
    return template_matrix, matching_library
    
    