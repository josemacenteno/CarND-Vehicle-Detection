**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[hog_o1]: ./output_images/hog/original_test1.jpg "Original test1"
[hog_t1]: ./output_images/hog/hog_test1.jpg      "hog test1"

[win_1]: ./output_images/win_search/test1.jpg "Sliding window on test1"
[win_small_1]: ./output_images/win_search/small_test1.jpg      "Small sliding window on test1"

[data_car_0]: ./data/vehicles/GTI_Far/image0000.png "car data 0"
[data_car_1]: ./data/vehicles/GTI_Left/image0009.png "car data 1"
[data_car_2]: ./data/vehicles/GTI_Left/image0010.png "car data 2"
[data_car_3]: ./data/vehicles/GTI_MiddleClose/image0000.png "car data 3"
[data_car_4]: ./data/vehicles/GTI_Right/image0000.png "car data 4"

[data_notcar_0]: ./data/non-vehicles/Extras/extra1.png "notcar data 0"
[data_notcar_1]: ./data/non-vehicles/Extras/extra100.png "notcar data 1"
[data_notcar_2]: ./data/non-vehicles/Extras/extra2.png "notcar data 2"
[data_notcar_3]: ./data/non-vehicles/Extras/extra3.png "notcar data 3"
[data_notcar_4]: ./data/non-vehicles/Extras/extra4.png "notcar data 4"

[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][data_car_0]        ![alt text][data_notcar_0]

![alt text][data_car_1]        ![alt text][data_notcar_1]

![alt text][data_car_2]        ![alt text][data_notcar_2]

![alt text][data_car_3]        ![alt text][data_notcar_3]

![alt text][data_car_4]        ![alt text][data_notcar_4]


I experimented on the quiz from class to choose the color space YCrCb, as it gave the best results measured by the SVC model accuracy on the a random training set. 

In the quiz and the project I used  `skimage.feature.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I started form the parameters that worked best in the quizes and turns out those gave good enough values for most cars. These are the three parameters used during training and pipeline:

```
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
```

I also used histogram and spatial features as suggested from the quizes and excercises. I used the X_scaler from sklearn to balance the heterogeneous nature of the aggregated hog, histogram and hog features.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The tagets are defined in the `train.py` file line 107 as 1 for cars and 0 for not_cars. The set of features and targets is split using train_test_split from sklearn. 

Finally the actual training of a linear classifier is defined in `train.py` between the lines 119 and 125.

Tha code after the fitting the classifier simply stores the classifier state in a pickle file, together with all the parameters used to define the "feature extraction" and the scaler. A pickle file is the recommended way to store a trained classifier according to sklearn documentation. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The main code to ferform the sliding window search is in the function `find_car` in `pipeline.py` lines 38 to 85. The image is first cropped by the ystart and ystop parameters, then resized based on a scaling factor. The scale fator implies a different window size with respect to the original image. In the main pipeline code I make two searches with find_car(), one for small windows in the center of the image (where far away cars would appear, and another search with larger windows on the bottom half of the image, which is better to identify close, therefore large, cars.

There windows advance through the image with a 3/4 overlap over the closest window.  Here is an illustration of the window size used and the grid searched for each of the scales applied:

![alt text][win_1]


![alt text][win_small_1]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

#TODO: Replace with actual images from my video pipeline

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I kept the last 25 bounding boxes from find_car() searches. A queue of size 25 was used, it is initialized in `pipeline.py` line 184.

This queue is used when a parameter persistance=True is passed to the pipeline method. This allows me to ignore previous detection when I am processing individual images as oposed to video sequences. 

If persistance is activated the last 25 detections will be combined with the new detections to make a heat map. From the heat map a threshold is used equal to the number of detections already in the queue.

Finally the heatmap is converted into single labels using the `label()` method form the scipy library. The labels are drawn onto the images and this is the final result of the pipeline.

#TODO: replace with my own examples.
### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I believe I could improve the results by using overlapping windows and maybe one more scale during the search. This would make the processing slower though. 

I can also play more with the C parameter in the Linear classifier, or try other classifiers.

The pipeline may fail in a more curvy road or one with steep ramps, as the search might not correspond to the new conditions. 

My false detections method is pretty fragile, and it was optimized for this specific project video. The false detection rejection can be improved by keeping track of individual cars and the time each car has beeen in sight. A weak detection in an area where a car has been detected for many frames is probably OK, but a weak detection where there hasn't been other cars before is more likely to be a false positive.

If I pursue this project further I would test with other videos and experiment using a CNN or RNN as the classifier.
