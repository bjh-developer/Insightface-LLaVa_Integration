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
import spacy
import csv
from ner_step import *
from LLaVa_step import *
from get_img_info import *
from insightface_temp import *
from insightface_test import *



count = 0


def full_programme(scrapedImages_cleaned, temp_img_list, matching_library, named_dictionary, template_matrix):

    ''' 
    full_programme function takes in a list of images utilises insightface_temp, insightface_test and llava_step functions
    to cluster images according to predicted name of the person in the image. 
    
    '''
    
    global count
    
    for img in scrapedImages_cleaned:
    
        count += 1
    
        if len(named_dictionary) == 0: #if named_dictionary contains no data, send the image through LLaVa and create a new key-value pair in the dictionary for it
            template_matrix, matching_library = insightface_temp(img, matching_library, template_matrix)
            lst = []
            lst.append(img)
            named_dictionary[count] = lst
            print(img, count)
            
        else:
            #checks if the person in the image matches with other images that are already in the named_dictionary database
            test, correct_temp_imgPerson_num, check, named_dictionary = insightface_test(img, template_matrix, matching_library, named_dictionary)
            print(img, count)
            if check == False:
                template_matrix, matching_library = insightface_temp(img, matching_library, template_matrix)
                lst = []
                lst.append(img)
                named_dictionary[count] = lst
                print(img, count, 'no match')
    
    
    print(named_dictionary)
    clusters = {'no name detected':[]}
    count = 0
    
    
    for num in named_dictionary:
        count += 1
        name = llava_step(named_dictionary[num][0]).rsplit("/")[0]
        
        if name.endswith(' '):
            name = name[:len(name)-1]
            
        if name == 'no name detected':
            for i in named_dictionary[num]:
                clusters['no name detected'].append(i)
            print(count)
            
        elif name not in clusters:
            clusters[name] = named_dictionary[num]
            print(count, name)
            
        else:
            for i in named_dictionary[num]:
                clusters[name].append(i)
            print(count, name)
        
            
    return clusters
                
