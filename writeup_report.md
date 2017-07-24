# **Behavioral Cloning**

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results
* run_submit.mp4 video for run on track 1

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model use NVIDIA model with 5 layers of convolution neural network.
This use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers. The model includes RELU layers to introduce nonlinearity and use dropout to reduce overfitting.
![GitHub Logo](/images/arch.png)

At the same time, the input image is normalized to [-1, 1] range and we chop the top 50 pixels and bottom 20 pixels to let the model only focus on the key monitor area.


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting

The model was trained and validated on different data sets to ensure that the model was not overfitting and we shuffled the input data before we split the examples. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Initially, I use 0.001 as start with 3 epochs for training and the car does not drive well, with 0.0001 learning rate and 10 epochs. It performs much better.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving. I use the basic data I got from the class and manually running the simulator to collect data with 2 path for original direction and 1 track of reverse direction. I also collect lane saver so it will save it when it is on edge. For the right turn and the exit after the bridge, there is limited data for such cases, I have add such example for the model to learn.
F

```
("data/driving_log.csv", 'data'),
("collected_data/driving_log.csv", ''),
("curve_saver/driving_log.csv", ''),
("reverse_lane/driving_log.csv", ''),
("speicial_edge/driving_log.csv", ''),
("right_turn/driving_log.csv", ''),
("low_resolution/driving_log.csv", ''
```


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving this model is try and error. We started with basic 1  convolution layer and one dense fully connected layer. The car drive randomly.

Then I use 2 convolution layer and 1 fully connected layer and crop the image as the guide mentioned. The car is able to drive straight and only make small turns.

Then I started to use the NVIDIA pipeline and it works much better. It took longer time to train as well. The car is driving ok until the exit after the bridge and the first right turn.

To overcome this, I generated more data with simulator and add speical layer for the that exit and right turn. At the same time, we lower down the learning rate from 0.001 to 0.0001 and change the epochs from 3 to 10. The car is performing much better with that.

For the input data, I am using left, right and center images for training. For left and right camera, I use small correction of 0.2. For center image, I flipped the image horizontally and change the steering as well to have more training data.

To overcome overfitting, I split the data and use 20% of the data as validation data set. I use dropout in fully connected layer to help the overfitting.

After all of these, the car is able to driving smoothly in track one. It still fails on track two since no track two data is applied.


#### 2. Final Model Architecture

My model use NVIDIA model with 5 layers of convolution neural network.
This use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers. The model includes RELU layers to introduce nonlinearity and use dropout to reduce overfitting.
![GitHub Logo](/images/arch.png)

At the same time, the input image is normalized to [-1, 1] range and we chop the top 50 pixels and bottom 20 pixels to let the model only focus on the key monitor area.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![image normal1](/images/normal1.jpg)
![image normal2](/images/normal2.jpg)
![image normal2](/images/normal3.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to These images show what a recovery looks like starting :

![image curve_saver](/images/curve_saver1.jpg)
![image curve_saver](/images/curve_saver2.jpg)
![image curve_saver](/images/curve_saver3.jpg)

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![image curve_saver](/images/image_original1.png)
![image curve_saver](/images/image_flip1.png)

After the collection process, I had 15905 entries, and 63620 number of data points. I then preprocessed this data by this architecture describe above.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.  I used an Adam optimizer so that manually training the learning rate wasn't necessary. I started the learning rate 0.001 and 3 epochs, the car does good in most part, wither lower learning rate to 0.0001 and 10 epochs, it performing much better.
