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


def insightface_test(test, template_matrix, matching_library, named_dictionary):
    '''
    insightface_test function takes in a test image and checks it against the template_matrix compiled from template images
    and returns True if the image matches, vice-versa.
    
    '''
    
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
    
    
    #determines if test image is similar to template image with the similarity score threshold set to 0.6
    for i in test_img_path:
        if max_sim_list[test_img_path.index(i)] >= 0.6:
            value_to_check = matching_library[template_match_list[0]]
            for key, sublist in named_dictionary.items():
                if value_to_check in sublist:
                    correct_temp_imgPerson_num = key
            named_dictionary[correct_temp_imgPerson_num].append(test)
            check = True
            return True, correct_temp_imgPerson_num, check, named_dictionary
            
            
    return False, None, False, named_dictionary
    
       