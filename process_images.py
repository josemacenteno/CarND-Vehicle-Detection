#!/home/jcenteno/miniconda3/envs/carnd-term1/bin/python
import cv2
import glob
from pipeline import *

# #Undistort all test images
# print("Extracting features from hog, histogram and spatial features")
# image_name = glob.glob('./data/vehicles/GTI_Far/image0000.png'):
# image = cv2.imread(image_name)    
# feature_image = convert_color(image, conv='BGR2' + color_space)

# # Call get_hog_features() with vis=True, feature_vec=True
# hog0, hog_img0 = get_hog_features(feature_image[:,:,0], 
#                                   orient, pix_per_cell, cell_per_block, 
#                                   vis=True, feature_vec=True)
# # Call get_hog_features() with vis=True, feature_vec=True
# hog1, hog_img2 = get_hog_features(feature_image[:,:,1], 
#                                   orient, pix_per_cell, cell_per_block, 
#                                   vis=True, feature_vec=True)
# # Call get_hog_features() with vis=True, feature_vec=True
# hog2, hog_img2 = get_hog_features(feature_image[:,:,2], 
#                                   orient, pix_per_cell, cell_per_block, 
#                                   vis=True, feature_vec=True)


# hog_features = np.ravel(hog_features)   






# small = cv2.resize(image,(256, 144))
# small_u = cv2.resize(undistorted,(256, 144))

# output_name = "./output_images/camera_calibration/original_" + image_name.split('/')[-1]
# cv2.imwrite(output_name, small)
# output_name = "./output_images/camera_calibration/undistorted_" + image_name.split('/')[-1]
# cv2.imwrite(output_name, small_u)

# output_name = "./output_images/camera_calibration/" + image_name.split('/')[-1]
# cv2.imwrite(output_name, undistorted)
# cv2.imshow('img',undistorted)
# cv2.waitKey(50)


#HOG
# print("Drawing detected windows on a test image 1")
# for image_name in glob.glob('./test_images/test*.jpg'):
#     print(image_name)
#     image =  cv2.imread(image_name) 
#     draw_img = image.copy()
#     in_color_channel = 'BGR'
#     windows = find_cars(image, in_color_channel, ystart, ystop, scale, svc, X_scaler, orient,
#                         pix_per_cell, cell_per_block, color_space, spatial_size, hist_bins) 
#     small_windows = find_cars(image, in_color_channel, ystart_small, ystop_small, scale_small,
#                         svc, X_scaler, orient, pix_per_cell, cell_per_block, color_space,
#                         spatial_size, hist_bins)  

#     # Draw the boxes on the image
#     for bbox in windows + small_windows:
#         cv2.rectangle(draw_img, bbox[0], bbox[1], (255,0,0), 6)

#     small = cv2.resize(cv2.imread(image_name),(256, 144))
#     small_p = cv2.resize(draw_img,(256, 144))
#     output_name = "./output_images/detected/original_" + image_name.split('/')[-1]
#     cv2.imwrite(output_name, small)
#     output_name = "./output_images/detected/detected_" + image_name.split('/')[-1]
#     cv2.imwrite(output_name, small_p)
#     output_name = "./output_images/detected/" + image_name.split('/')[-1]
#     cv2.imwrite(output_name, draw_img)

#     cv2.imshow('img',draw_img)
#     cv2.waitKey(50)






##Illustrate sliding window search
print("Drawing search windows on a test image 1")
image_name = './test_images/test1.jpg'
print(image_name)
image =  cv2.imread(image_name) 
draw_img = image.copy()
draw_img_small = image.copy()
in_color_channel = 'BGR'
windows = find_cars(image, in_color_channel, ystart, ystop, scale, svc, X_scaler, orient,
                    pix_per_cell, cell_per_block, color_space, spatial_size, hist_bins, return_all_windows = True) 
small_windows = find_cars(image, in_color_channel, ystart_small, ystop_small, scale_small,
                    svc, X_scaler, orient, pix_per_cell, cell_per_block, color_space,
                    spatial_size, hist_bins, return_all_windows = True)  

# Draw the boxes on the image
for bbox in windows:
    cv2.rectangle(draw_img, bbox[0], bbox[1], (255,0,0), 6)
cv2.rectangle(draw_img, bbox[0], bbox[1], (0,155,0), 6)

# Draw the small boxes on the image
for bbox in small_windows:
    cv2.rectangle(draw_img_small, bbox[0], bbox[1], (255,0,0), 6)
cv2.rectangle(draw_img_small, bbox[0], bbox[1], (0,155,0), 6)


cv2.imshow('img',draw_img)
cv2.waitKey(500)
cv2.imshow('img',draw_img_small)
cv2.waitKey(500)

output_name = "./output_images/win_search/" + image_name.split('/')[-1]
cv2.imwrite(output_name, draw_img)

output_name = "./output_images/win_search/small_" + image_name.split('/')[-1]
cv2.imwrite(output_name, draw_img_small)


#Detections
print("Drawing detected windows on a test image 1")
for image_name in glob.glob('./test_images/test*.jpg'):
    print(image_name)
    image =  cv2.imread(image_name) 
    draw_img = image.copy()
    in_color_channel = 'BGR'
    windows = find_cars(image, in_color_channel, ystart, ystop, scale, svc, X_scaler, orient,
                        pix_per_cell, cell_per_block, color_space, spatial_size, hist_bins) 
    small_windows = find_cars(image, in_color_channel, ystart_small, ystop_small, scale_small,
                        svc, X_scaler, orient, pix_per_cell, cell_per_block, color_space,
                        spatial_size, hist_bins)  

    # Draw the boxes on the image
    for bbox in windows + small_windows:
        cv2.rectangle(draw_img, bbox[0], bbox[1], (255,0,0), 6)

    output_name = "./output_images/detected/" + image_name.split('/')[-1]
    cv2.imwrite(output_name, draw_img)

    cv2.imshow('img',draw_img)
    cv2.waitKey(50)

#Heat map
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, windows + small_windows)

    # Visualize the heatmap when displaying 
    heatmap = 1/10*heat
    cv2.imshow('img',heatmap)
    cv2.waitKey(50)

    # Find final boxes from heatmap using label function
    labels = label(heat)  
    lablesmap = labels[0]/2

    print(np.amax(lablesmap))
    cv2.imshow('img',lablesmap)
    cv2.waitKey(50)

    small = cv2.resize(image,(256, 144))
    small_hm = cv2.resize(heatmap*255,(256, 144))
    small_l = cv2.resize(lablesmap*255,(256, 144))
    small_d = cv2.resize(draw_img,(256, 144))
    output_name = "./output_images/detected/original_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small)
    output_name = "./output_images/detected/heatmap_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small_hm)
    output_name = "./output_images/detected/label_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small_l)
    output_name = "./output_images/detected/detected_" + image_name.split('/')[-1]
    cv2.imwrite(output_name, small_d)

#     cv2.imshow('img',draw_img)
#     cv2.waitKey(50)


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
