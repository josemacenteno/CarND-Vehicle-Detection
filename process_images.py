#!/home/jcenteno/miniconda3/envs/carnd-term1/bin/python
import cv2
import glob
from pipeline import *

#Undistort all test images
print("Extracting features from hog, histogram and spatial features")
image_name = glob.glob('./data/vehicles/GTI_Far/image0000.png'):
image = cv2.imread(image_name)    
feature_image = convert_color(image, conv='BGR2' + color_space)

# Call get_hog_features() with vis=True, feature_vec=True
hog0, hog_img0 = get_hog_features(feature_image[:,:,0], 
                                  orient, pix_per_cell, cell_per_block, 
                                  vis=True, feature_vec=True)
# Call get_hog_features() with vis=True, feature_vec=True
hog1, hog_img2 = get_hog_features(feature_image[:,:,1], 
                                  orient, pix_per_cell, cell_per_block, 
                                  vis=True, feature_vec=True)
# Call get_hog_features() with vis=True, feature_vec=True
hog2, hog_img2 = get_hog_features(feature_image[:,:,2], 
                                  orient, pix_per_cell, cell_per_block, 
                                  vis=True, feature_vec=True)


hog_features = np.ravel(hog_features)   






small = cv2.resize(image,(256, 144))
small_u = cv2.resize(undistorted,(256, 144))

output_name = "./output_images/camera_calibration/original_" + image_name.split('/')[-1]
cv2.imwrite(output_name, small)
output_name = "./output_images/camera_calibration/undistorted_" + image_name.split('/')[-1]
cv2.imwrite(output_name, small_u)

output_name = "./output_images/camera_calibration/" + image_name.split('/')[-1]
cv2.imwrite(output_name, undistorted)
cv2.imshow('img',undistorted)
cv2.waitKey(50)

#Aplying pipeline to all test images
print("Applying pipeline to test images from ./test_images/...")
for image_name in glob.glob('./test_images/test*.jpg'):
    print(image_name)
    image =  cv2.imread(image_name) 
    pipelined = pipeline(image, persistance=False, in_color_channel = 'BGR')
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
