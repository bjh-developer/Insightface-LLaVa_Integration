from LLaVa_step import *
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def majority_voting(clusters, data):

    '''
    majority_voting function takes in the clusters of images with its predicted name attcahed to it,
    runs LLaVa on all the images under a specific individual,
    if there's only one image attached to the person's name, mark it as 'correctly named',
    if there's only two images, get the predicted names of both images and use fuzzywuzzy to check similarity of the two names,
    if there's more than two images, get the predicted names of all the images and run it through majority voting to determine if the images under the person's name are reliably named.
        
    '''


    for person in clusters.copy():
        print(person)
        if (person != 'no name detected'):
            print(len(clusters[person]))
            if (len(clusters[person]) == 1):
                data['correctly named'] += len(clusters[person])
            else:
                names = [person]
                photos = [clusters[person][0]]
                count = 1
                for photo in clusters[person]:
                    if count > 1:
                        name = llava_step(photo).rsplit("/")[0]
                        if name != 'no name detected':
                            names.append(name)
                            photos.append(photo)
                    count += 1
                        
                if len(names) == 2:
                    # Gets similarity score of the 2 predicted names from the images
                    sim_score = fuzz.ratio(names[0], names[1])
                    print(sim_score)
                    if sim_score < 50:
                        for i in clusters[person]:
                            clusters['no name detected'].append(i)
                        data['wrongly named'] += len(clusters[person])
                        clusters.pop(person)
                else:
                    # Majorty voting algorithm
                    candidate = None
                    votes = 0
                    for i in range(len(names)):
                        if votes == 0:
                            candidate = names[i]
                            votes = 1
                        else:
                            if names[i] == candidate:
                                votes += 1
                            else:
                                votes -= 1
                    count = 0
                    
                    # Checking if majority candidate occurs more than len(names)/2 times
                    for i in range(len(names)):
                        if names[i] == candidate:
                            count += 1
                    if (count > len(names) // 2):
                        data['correctly named'] += len(clusters[person])
                    else:
                        for i in clusters[person]:
                            clusters['no name detected'].append(i)
                        data['wrongly named'] += len(clusters[person])
                        clusters.pop(person)
            
        else:
            data['no name detected'] += len(clusters['no name detected'])
               
    return clusters, data

