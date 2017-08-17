#**Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:(due to size constraint in github, model.h5 was removed)
* model.py containing the script to create and train the model
* data folder https://drive.google.com/file/d/0B--NLNkfW8vnVDRqLTNZNlQ2aGM/view?usp=sharing
* run3.mp4 for video of automous driving on track
* writeup.md summarizing the results

####2. Submission includes functional code

model.py will train model based on data collected in data/ folder
run3.mp4 captures the autonomous driving results via running model.h5

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

The model.py contains preprocessing, modified CNN layer based on Nvidia model

####1. An appropriate model architecture has been employed

model.py line 87 modified_nvida_model shows how model architecture was employed

The model consists with 
* normalization line 91 and croping line 92 as start witch generate (160, 320, 3) matrix with cell value [-0.5,0.5]
* four layers of convolution layers with a MaxPooling line 93 - line 98
* fully connected layers line 101 - 108 with dropouts 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 101-107). 

The model was trained and validated on randomized data sets to ensure that the model was not overfitting (code line 57). 
The left,center,right was randomized line 61 picked and brightness was randomized line 72. The image is fliped 50% to oposite 
direction line 76.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, learning rate tuned to 0.0001 (model.py line 132).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, recovering from the left and right sides of the road
I also pick training data in reverse turns

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to mimic what nvidia model does

My first step was to use a convolution neural network model similar to the nvida model I thought this model might be appropriate because it tackles very similar problem

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I add drop outs to fully connected layers

Then I still see overfiting and bias towards driving stright. I visualize the data sample and find heavy lean toward 0 steering.
So I decided to drop some samples and fix traning set steering distribution. Then I also does utilize both left, right and center image 
to randomize image inputs, then I do flip image and do more reverse driving to fix training data set left turn bias.

At last, I did crop and normalize image in model

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I add more trianing
set just doing sharp turns when it almost hit the border(e.g bridge)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. run3.mp4
