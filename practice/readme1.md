**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell(get_hog_features()) of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=16`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters to find the maximum accuracy. Below are some of the combinations that I tried.

##### Using: Linear kernel,  RGB color_space, 8 orientations, 16 pixels per cell, 2 cells per block, 0 hog_channel and (16,16) spatial size
Feature vector length: 1104
7.38 Seconds to train SVC...
Test Accuracy of SVC =  0.9609

##### Using: Linear kernel, RGB color_space, 8 orientations, 8 pixels per cell, 2 cells per block, 0 hog_channel and (32,32) spatial size
Feature vector length: 4688
22.65 Seconds to train SVC...
Test Accuracy of SVC =  0.9535

##### Using: Linear kernel, RGB color_space, 8 orientations, 8 pixels per cell, 2 cells per block, ALL hog_channel and (24,24) spatial size
Feature vector length: 6480
31.14 Seconds to train SVC...
Test Accuracy of SVC =  0.9617

##### Using: Linear kernel, RGB color_space, 9 orientations, 16 pixels per cell, 1 cells per block, 1 hog_channel and (24,24) spatial size
Feature vector length: 2100
10.93 Seconds to train SVC...
Test Accuracy of SVC =  0.9558

##### Using: Linear kernel, RGB color_space, 9 orientations, 16 pixels per cell, 2 cells per block, ALL hog_channel and (32,32) spatial size
Feature vector length: 4092
20.69 Seconds to train SVC...
Test Accuracy of SVC =  0.9603

##### Using: RBF kernel,  YCrCb color_space, 16 orientations, 16 pixels per cell, 2 cells per block, 0 hog_channel and (16,16) spatial size
Feature vector length: 1392
60.3 Seconds to train SVC...
Test Accuracy of SVC =  0.9913

Although RBF kernel took a lot of time to train the SVM but the result was fruitfull. My final accuracy was 99.13 %.

####  3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

To train the SVM first I extracted the features out of cars and not cars images. The code to extract images is present in extract_features method. Then I splitted up the data in to training and validation set. Then I used the training data to train SVM using RBF kernel. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decied to use 3 sliding windows of size (64, 64), (96,96), and (128,128) on lower half of  the image with overlapping of 75% . Since I am using hog features with a threshold of 2, so to make sure the presence of car on a particualr region I slided all three windows on this area. 

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 1-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

