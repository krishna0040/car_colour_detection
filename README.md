# car_colour_detection
Detecting the colour car and any presence of pedestrains  

#### Download the model using the link given in model
Run the training file on colab by selecting runtype - gpu (total run time approx 20min)

We use the vcor dataset from kaggle and combine the train, test and validation data. Also resize the image to 128,128 pixels and scale the images. 
Now delete the unnecessary vairables and then split the data into train and test. To reduce the usage of ram (which is limited by 15 Gb), convert the data to Tensorflow Dataset, which used the gpu gpu ram.
Now build the model, so that it does now overfit the training data. The data is skewed as we have less images of blue and more of the others, so i assigned wieghts while training the model. Random oversampling or SMOT could also be used, but it caused the model to overfit the blue images. 

Then apply Image Augmentation, like rotate , flip, zoom in, zoom out and other image operations, to furhter prevent overfitting of the model. The train the model, such that if the loss does not improve for 30 epochs, we would stop training or if the loss does not improve for 3 epochs, reduce the learning rate. After training the model we can set a threshold probability for prediction of blue according to our choice. I set the probability  to 0.65 so that only cars with surity of blue are selected. Also at threshold =0.5, sometimes purple and other colours similar to blue were wronlgy predicted to be blue.

From the classification report and confusion matrix, we can see that the model has performed very well on the test data and from the preview images, we can see that the model again performs very well.

We can improve the detection of cars by using a yolov8-s or yolov8-m, but it will take more time for computation.

The gui draws a bounding box around the detected car from the uploaded image, when detect is pressed. A red box is drawn for blue cars, blue box for other coloured cars and a green box for pedestrians.

