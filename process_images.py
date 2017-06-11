#!/home/jcenteno/miniconda3/envs/carnd-term1/bin/python
import cv2
import glob
from pipeline import *

#Read images in RGB format
def rgb_read(path):
    image_bgr = cv2.imread(path)  
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

#Aplying pipeline to all test images
print("Applying pipeline to test images from ./test_images/...")
for image_name in glob.glob('./test_images/test*.jpg'):
    image =  rgb_read(image_name) 
    pipelined_rgb = pipeline(image, persistance=False)

    pipelined = cv2.cvtColor(pipelined_rgb, cv2.COLOR_RGB2BGR)
    small = cv2.resize(cv2.imread(image_name),(256, 144))
    small_p = cv2.resize(pipelined,(256, 144))

    output_name = "./output_images/pipelined/original_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small)
    output_name = "./output_images/pipelined/piped_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small_p)

    output_name = "./output_images/pipelined/" + image_name.split('/')[-1]
    cv2.imwrite(output_name, pipelined*255)

    cv2.imshow('img',pipelined)
    cv2.waitKey(50)
