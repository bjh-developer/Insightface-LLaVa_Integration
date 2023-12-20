# Run this code to execute the entire programme (faster version)

import random
import cv2
from simple_image_download import simple_image_download as simp
import insightface
import time
import os
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from full_programme import *
from accuracy_check import *
from majority_voting import *


#creating dictionary to store data values to plot bar graph to profile accuracy of programme
data = {'correctly named':0, 'wrongly named':0, 'no name detected':0}


#adding all the images into scrapedImages variable and creating scrapedImages_cleaned variable to store cleaned images (images with faces inside)
mypath = "/data2/Intern/Insightface-LLaVa/Scraped_images/"
scrapedImages = [f for f in listdir(mypath) if isfile(join(mypath, f))]
scrapedImages_cleaned = []

temp_img_list = []
matching_library = {}
named_dictionary = {}
template_matrix = None


#asks if user wants data to be cleaned to ensure that it only runs on images with faces in it
userCleanData = input('clean data [y/n]: ')


if userCleanData.lower() == 'y':
    count = 0
    
    #runs through all the images in the scrapedImages folder to check if there are faces
    for i in scrapedImages:
        count += 1
        print(i, count)
        img = cv2.imread(mypath + i)
        faces = app.get(img)
        if len(faces) != 0:
            scrapedImages_cleaned.append(i)
    print(len(scrapedImages_cleaned))
    
    #main programme to sort images into its relevant human name clusters
    random.shuffle(scrapedImages_cleaned)
    start = time.time()
    clusters = full_programme(scrapedImages_cleaned, temp_img_list, matching_library, named_dictionary, template_matrix)
    end = time.time()
    print(end-start)
    for x in clusters:
        print(x, clusters[x])
    
    print(clusters)
    print(data)
    
    #checks if all the images under a specific individual are the same through majority voting algorithm
    clusters, data = majority_voting(clusters, data)
    
    print(clusters)
    print(data)
  
    
    #checks if the person in the image is correctly named based on Google search
    #getting template images (from google) of the people identified in the images from above
    response = simp.simple_image_download
    lst = []
    for name in clusters:
        if name != 'no name detected':
            lst.append(name)
    mypath = "/data2/Intern/Insightface-LLaVa/simple_images/"
    check_files = os.listdir(mypath)
    for rep in lst:    
        if rep not in check_files:
            print(f'downloading {rep}')
            response().download(rep, 20)
            print('done')
        else:
            mypath = '/data2/Intern/Insightface-LLaVa/simple_images/' + rep
            check_file_numbers = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
            check_for = rep + '_20.jpg'
            if check_for not in check_file_numbers:
                print(f'downloading {rep}')
                response().download(rep, 20)
                print('done')
            else:
                print(f'{rep} already in database')
                
    for names in lst:
        data, clusters = accuracy_check(names, clusters, data)
    
    for i in clusters:
        print(i, clusters[i])
    print(data)
    
    #creating bar plot graph from data returned by accuracy_check function    
    labels = list(data.keys())
    values = list(data.values())
    fig, ax = plt.subplots()
    bar_container = ax.bar(labels, values)
    ax.set(ylabel='Number of images', title='Insightface-LLaVa Classification')
    ax.bar_label(bar_container, fmt='{:,.0f}')
    plt.savefig('insightface-llava_performance.png')
      

elif userCleanData.lower() == 'n':
    
    #main programme to sort images into its relevant human name clusters
    start = time.time()
    clusters = full_programme(scrapedImages_cleaned, temp_img_list, matching_library, named_dictionary, template_matrix)
    end = time.time()
    print(end-start)
    
    for x in clusters:
        print(x, clusters[x])
    
    print(clusters)
    print(data)
    
    #checks if all the images under a specific individual are the same through majority voting algorithm
    clusters, data = majority_voting(clusters, data)
    
    print(clusters)
    print(data)
    
    
    #checks if the person in the image is correctly named based on Google search
    #getting template images (from google) of the people identified in the images from above
    response = simp.simple_image_download
    lst = []
    for name in clusters:
        if name != 'no name detected':
            lst.append(name)
    mypath = "/data2/Intern/Insightface-LLaVa/simple_images/"
    check_files = os.listdir(mypath)
    for rep in lst:    
        if rep not in check_files:
            print(f'downloading {rep}')
            response().download(rep, 20)
            print('done')
        else:
            mypath = '/data2/Intern/Insightface-LLaVa/simple_images/' + rep
            check_file_numbers = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
            check_for = rep + '_20.jpg'
            if check_for not in check_file_numbers:
                print(f'downloading {rep}')
                response().download(rep, 20)
                print('done')
            else:
                print(f'{rep} already in database')
                
    for names in lst:
        data, clusters = accuracy_check(names, clusters, data)
    
    for i in clusters:
        print(i, clusters[i])
    print(data)
    
    #creating bar plot graph from data returned by accuracy_check function    
    labels = list(data.keys())
    values = list(data.values())
    fig, ax = plt.subplots()
    bar_container = ax.bar(labels, values)
    ax.set(ylabel='Number of images', title='Insightface-LLaVa Classification')
    ax.bar_label(bar_container, fmt='{:,.0f}')
    plt.savefig('insightface-llava_performance.png')
        
        
else:
    print('end')
    
