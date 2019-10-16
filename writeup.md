# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/bar_char.jpg "Visualization"
[image2]: ./examples/gray_traffic.jpg "Grayscaling"
[image4]: ./my_images/test1.jpg "Traffic Sign 1"
[image5]: ./my_images/test2.jpg "Traffic Sign 2"
[image6]: ./my_images/test3.jpg "Traffic Sign 3"
[image7]: ./my_images/test4.jpg "Traffic Sign 4"
[image8]: ./my_images/test5.jpg "Traffic Sign 5"

[image9]: ./my_images/char1.jpg "Speed limit (120km/h)"
[image10]: ./my_images/char2.jpg "Speed limit (30km/h)"
[image11]: ./my_images/char3.jpg "Right-of-way at the next intersection"
[image12]: ./my_images/char4.jpg "Turn right ahead"
[image13]: ./my_images/char5.jpg "Stop"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I find some class bias issue as some classes seem to be underrepresented.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I shuffeled the data.

Then, I decided to convert the images to grayscale because the accuracy of the model is higher when doing so.

Here is an example of a few traffic sign image after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because CNNs work better when the distance between the diffrent data points is rather small then large (magnitude of the pixel values).


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the a modified version of the VGGNet, which won the ImageNet Large Scale Visual Recognition Competition in 2014:
https://medium.com/coinmonks/paper-review-of-vggnet-1st-runner-up-of-ilsvlc-2014-image-classification-d02355543a11

The final model consisted of the following layers. I found this modifiey version after I did some research about the VGGNet:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU                  | 	     										|
| Convolutional 3x3     |1x1 stride, same padding, outputs 32x32x32		|
| RELU                  | 	     										|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x64	|
| RELU                  | 	     										|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x64	|
| RELU                  | 	     										|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 					|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 8x8x128		|
| RELU                  | 	     										|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 8x8x128		|
| RELU                  | 	     										|
| Max pooling	      	| 2x2 stride,  outputs 4x4x128 					|
| Fully connected		| Output 2048        							|
| Fully connected		| Output 128     								|
| RELU                  | 	     										|
| Fully connected		| Output 43 = logits     						|
 


#### 3. Describtion of the model: Batch size, number of epochs and any hyperparameters

To train the model, I used an batch size of 64 and 10 epochs, in combination with a small learnign rate of 0.001.

#### 4. Discussion

My final model results were:
* validation set accuracy of 0.989
* test set accuracy of 0.967

An iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first architecture I choose was the well known LeNet architecture. However, the accuracy was mostly under 0.9. I choose this model since it is well known.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I triede to improve the architecture by try and error and got a better feeling of what can improve my accuracy. Nevertheless, I did not achive to get a better accuracy than 0.93. So I choose a well known architecture to complete this project.


If a well known architecture was chosen:
* What architecture was chosen?
I used the a modified version of the VGGNet, which won the ImageNet Large Scale Visual Recognition Competition in 2014:
https://medium.com/coinmonks/paper-review-of-vggnet-1st-runner-up-of-ilsvlc-2014-image-classification-d02355543a11
* Why did you believe it would be relevant to the traffic sign application?
Since it won the Large Scale Visual Recognition Competition it seemed like a pretty good approach. After some research I used a modified version of VGGNet.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
As I will oultine later the model is working well on the test set since the accuracy is high. However, some signs are underrepresented in the whole datasets. The CNN is not so good in predicting them.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Some images are really dificult to classify. For example the 30km/h sign has a black dot on the right of the "0", which is uncommun and should be hard to classify as well. However signs like the "Turn right ahead" and "STOP" sign should be easy to identify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			  						|Prediction										| 
|:-------------------------------------:|:---------------------------------------------:| 
| Speed limit (120km/h) 				| Speed limit (30km/h)  						| 
| Speed limit (30km/h)  				| Speed limit (80km/h) 							|
| Right-of-way at the next intersection	| Right-of-way at the next intersection			|
| Turn right ahead	 					| Turn right ahead				 				|
| Stop									| Stop											|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. In comparision to the test set this is not such a good accuracy. However, as I already mentioned above, some of the signs are hard to identify and 5 signs is not a significant sample. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

[alt text][image9]

For the first image, the model is relatively unsure. As you can see, it not even considers the 120 km/h sign to be under the top 5. However the top 2 options are speed limits which is correct.

[alt text][image10]
For the second image, the model is relatively unsure, as well. As you can see it is sure that its a speed limit but it is not able to identify which one. 

[alt text][image11]
Here the sign is correct. As you can see the CNN is very sure about it.

[alt text][image12]
The "Turn right ahead" sign is correct as well.

[alt text][image13]
The stop sign was not so easy to identify. The propablitiy for a speed limit is high as well. The reason for this might be that there is a huge number of speed limits in the training set and not as many stop signs.



