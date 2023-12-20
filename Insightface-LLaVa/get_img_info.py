import csv
from ner_step import *


def get_img_info(imgName):

    '''
    get_img_into function takes in the name of the web-scrapped image file, retrieves the title and caption of the web-scrapped image and returns it.
    In addition, it also attempts to retrieve possible names of the people in the images by using Named Entity Recognition using spaCy on the title and caption of the image.
        
    '''


    with open('cna.csv', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if imgName not in row[4]:
                info = {'title':'no title detected', 'caption':'no caption detected', 'names':'no name detected'}
            else:
                names_from_title = ner_step(row[0])
                names_from_caption = ner_step(row[2])
                if names_from_title == None and names_from_caption == None:
                    names = 'no name detected'
                elif names_from_title == None:
                    names = names_from_caption
                elif names_from_caption == None:
                    names = names_from_title
                else:
                    names = names_from_title + ' / ' + names_from_caption
                info = {'title':row[0], 'caption':row[2], 'names':names}
                break
                
    return info
    
    