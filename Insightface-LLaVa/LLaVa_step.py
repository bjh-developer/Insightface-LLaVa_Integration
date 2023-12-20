import os
os.environ['TRANSFORMERS_CACHE'] = '/data2/Intern/Cache/'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import argparse
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from io import BytesIO
from PIL import Image
from insightface_step import *
from get_img_info import *
from ner_step import *
import numpy as np
import pandas as pd
import requests
import cv2


disable_torch_init()


# Model Configs
class Model_Config():
    def __init__(self):
        self.temperature = 0.2
        self.max_new_tokens = 512
        self.image_aspect_ratio = 'pad'

model_path = 'liuhaotian/llava-v1.5-7b'
load_8bit = True
load_4bit = False
device = "cuda"
conv_mode = "llava_v1"
model_config = Model_Config()
max_text_length = 2048

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)


def caption_image(image_file, prompt):

    '''
    caption_image function enables us to give LLaVa an image and its prompt which will return LLaVa's generated output.

    '''

    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    inp = f"{roles[0]}: {prompt}"
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
      output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2,
                                  max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    output = outputs.rsplit('</s>', 1)[0]
    return image, output
    

def llava_step(image):

    '''
    llava_step function takes in an image and sends it to LLaVa model model to get a generated output from it, thereby returning predicted name of the person in the image.
    As the output from LLaVa model is a sentence replying to the prompt given, Named Entity Recognition by spaCy is used (ner_step function) to return only the names.
    
    '''


    mypath = "/data2/Intern/Insightface-LLaVa/Scraped_images/"
    img = cv2.imread(mypath + image)
    faces = app.get(img)
    info = get_img_info(image)
    
    
    if len(faces) == 1:
        image, output = caption_image(f'/data2/Intern/Insightface-LLaVa/Scraped_images/{image}', f"Title of image: {info['title']}. Caption of image: {info['caption']}. Potential names of the person in the image: {info['names']}. Given the title, caption of the image and potential names of the person in the image, who is in the centre of this image?")
        print(output)
        names = ner_step(output)
        if names == None:
            return 'no name detected'
        else:
            return names
            
            
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
        box = correct_face.bbox
        start = (int(box[0]),int(box[1]))
        end = (int(box[2]),int(box[3]))
        rimg = cv2.rectangle(img, start, end, (255,0,0), 2)
        cropped_img = rimg[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        
        if len(cropped_img) == 0:
            return 'no name detected'
        else:
            cv2.imwrite("/data2/Intern/Insightface-LLaVa/single_face_output.jpg", cropped_img)
        image, output = caption_image(f'/data2/Intern/Insightface-LLaVa/single_face_output.jpg', f"Title of image: {info['title']}. Caption of image: {info['caption']}. Potential names of the person in the image: {info['names']}. Given the title, caption of the image and potential names of the person in the image, who is in the image?")
        print(output)
        names = ner_step(output)
        
        if names == None:
            return 'no name detected'
        else:
            return names
            
            