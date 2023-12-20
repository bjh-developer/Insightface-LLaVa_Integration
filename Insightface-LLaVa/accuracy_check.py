from simple_image_download import simple_image_download as simp
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


def accuracy_check(name, clusters, data):

    '''
    accuracy_check function takes in name of preson in image, the full named_dictionary cluster of all images and data dictionary (containing 3 key-value pair to account for correctly and wrongly categorised images, and returns the updated data dictionary.
    It takes the template images of the named individual downloaded from Google in run.py and test the web-scrapped categorised images against it.
    If it matches, data['correctly named'] will remain the same (as given by majority voting).
    If it doesn't match, data['wrongly named'] will increment by 1, data['correctly named'] decrease by 1.
    If no name was detected, data['no name detected'] will increment by 1.
        
    '''

    
    #sieves through the downloaded Google images for a suitable template image to be used (i.e. only the targetted individual in the image)
    template_matrix = None
    done = False
    count = 20
    while done == False:
        if count == 0:
            return data, clusters
        
        path = '/data2/Intern/Insightface-LLaVa_V2.py/simple_images/' + name + '/' + name + '_' + str(count) + '.jpg'
        img = cv2.imread(path)
        if img is None:
            count -= 1
        else:
            faces = app.get(img)
            if len(faces) > 1:
                count -= 1
            elif len(faces) == 0:
                return data, clusters
            else:
                print(path)
                #takes downloaded image and generate a template from it
                face = faces[0]
                embed = face.normed_embedding
                embed = embed.reshape(1, -1)
                template_matrix = embed
                done = True
    

    max_sim_list = []
    template_match_list = []
    test_img_path = []
    onlyfiles = [f for f in clusters[name]]
    mypath = "/data2/Intern/Insightface-LLaVa/Scraped_images/"
    
    
    #test the scrapped categorised images against template image
    for file in onlyfiles:
        test_img_path.append(file)
        img = cv2.imread(mypath + file)
        faces = app.get(img)
        
        if len(faces) == 1:
            face = faces[0]
            embed = face.normed_embedding
            embed = embed.reshape(1, -1)
            
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
            
            
        sim = np.matmul(template_matrix, embed.transpose())
        max_sim_index = np.argmax(sim)
        max_sim = sim[max_sim_index]
        template_match_list.append(max_sim_index)
        max_sim_list.append(max_sim)

    print()
    #determines if test image is similar to template image with the similarity score threshold set to 0.341
    print(test_img_path, len(test_img_path))
    for i in test_img_path:
        if max_sim_list[test_img_path.index(i)] < 0.341:
            print(i, name, max_sim_list[test_img_path.index(i)])
            clusters[name].remove(i)
            data['wrongly named'] += 1
            data['correctly named'] -= 1
                
    return data, clusters

