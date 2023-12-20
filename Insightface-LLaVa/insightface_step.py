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


def insightface_step(template, test, template_matrix):

    '''
    insightface_step function takes in a template image and a test image to compare the test image against the template, and returns a boolean to express if test image matches template image.
    It will return True if it matches.
    Else it will return False.
    
    '''


    #generating template
    mypath = "/data2/Intern/Insightface-LLaVa/Scraped_images/"
    template_matrix = None
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


    #test process
    mypath = "/data2/Intern/Insightface-LLaVa/Scraped_images/"
    max_sim_list = []
    template_match_list = []
    test_img_path = []
    test_img_path.append(test)
    img = cv2.imread(mypath + test)
    faces = app.get(img)
    if len(faces) == 1:
        face = faces[0]
        embed = face.normed_embedding
        embed = embed.reshape(1, -1)
        sim = np.matmul(template_matrix, embed.transpose())
        max_sim_index = np.argmax(sim)
        max_sim = sim[max_sim_index]
        template_match_list.append(max_sim_index)
        max_sim_list.append(max_sim)
    elif len(faces) == 0:
        template_match_list.append(None)
        max_sim_list.append(0)
    else:
        highest_max_sim = 0
        for face in faces:
            embed = face.normed_embedding
            embed = embed.reshape(1, -1)
            sim = np.matmul(template_matrix, embed.transpose())
            max_sim_index = np.argmax(sim)
            max_sim = sim[max_sim_index]
            if max_sim > highest_max_sim:
                highest_max_sim = max_sim
        template_match_list.append(max_sim_index)
        max_sim_list.append(highest_max_sim)


    #determines if test image is similar to template image with the similarity score threshold set to 0.3
    for i in test_img_path:
        if max_sim_list[test_img_path.index(i)] >= 0.3:
            return True
        else:
            return False
            
            