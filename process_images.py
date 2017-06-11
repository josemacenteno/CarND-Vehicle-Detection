#!/home/jcenteno/miniconda3/envs/carnd-term1/bin/python
import cv2
import glob
from pipeline import *

#Aplying pipeline to all test images
print("Applying pipeline to test images from ./test_images/...")
for image_name in glob.glob('./test_images/test*.jpg'):
    image =  cv2.imread(image_name) 
    pipelined = pipeline(image, persistance=False)
    small = cv2.resize(cv2.imread(image_name),(256, 144))
    small_p = cv2.resize(pipelined,(256, 144))

    output_name = "./output_images/pipelined/original_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small)
    output_name = "./output_images/pipelined/piped_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small_p)

    output_name = "./output_images/pipelined/" + image_name.split('/')[-1]
    cv2.imwrite(output_name, pipelined)

    cv2.imshow('img',pipelined)
    cv2.waitKey(50)
