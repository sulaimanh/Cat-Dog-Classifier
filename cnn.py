# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
# We will use this to initialize our NN
from keras.models import Sequential
# We will use this to perform convolution
from keras.layers import Conv2D
# We will use this to perform the pooling step
from keras.layers import MaxPooling2D
# We will use this to perform the flattening step
from keras.layers import Flatten
# We will use this to add the fully connected layers in a classic ANN
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution
# Arguments
#   - 32 represents the number of filters we will have
#   - 3 represents the number of rows in the filter
#   - 3 represents the number of columns in the filter
#   - input_shape represents the number of dimensions of the image, and the size
#       3 represents RGB with a size of 64x64
#   - This is the activation function. We need to have non-linearity in our model, so that is why we use ReLu
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))


# Step 2 - Pooling
# Arguments:
#   - pool_size is the size of the pooling matrix
# The default stride is 1
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
# Arguments:
#   1st: This is the number of nodes in the hidden layer
#   2nd: Here, we are randomly initializing the weights close to zero. We iniatialze them with a uniform function
#   3rd: This is the activation function we want to choose in our hidden layer
#   4th: This is the number of nodes in the input layer, the number of independent variables.
classifier.add(Dense(units = 128, activation = 'relu'))
# Output
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Part 2 - Fitting the CNN to the images
# This will preprocess the images to prevent overfitting
# Data Augmentation will create many batches of our images, and in each batch it will apply some random transformations on a random selection of
#   our images, like rotating them, flipping them, shifting them, or even shearing them, and eventually what we'll get during the training is many 
#   more diverse images inside these batches, and therefore a lot more material to train.
# So, Image Augmentation is a technique that allows us to enrich our data set, our training set, without adding more images and therefore that
#   allows us to get good performance results with little or no overfitting, even with a small amount of images. 
# This is based on the flow from directory method
from keras.preprocessing.image import ImageDataGenerator

# Here, we apply some several transformations 
# rescale will rescale our pixel values between zero and one
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Here, we rescale the images of our test set .
test_datagen = ImageDataGenerator(rescale = 1./255)


# This will create the training set
# Arguments:
#   1st: This is where we get the images from
#   2nd: This is the size of your images that is expected in the CNN model.
#   3rd: This is the size of the batches in which some random sample of our images will be included and that contains the number of iamges that
#       will go through the CNN, after which the weights will be updated. 
#   4th: This says whether the class, dependent variable, is binary or has more than two categories.
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# This will create the test set.
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# This will fit the CNN to the training set, while also testing its performance on the test set.
# Remember that an epoch is a complete pass through the entire training data set.
# Arguments:
#   1st: the training set
#   2nd: Total number of steps (batches of samples) to yield from generator before declaring one epoch 
#       finished and starting the next epoch. It should typically be equal to the number of samples of 
#       your dataset divided by the batch size
#   3rd: This is the number of epochs you want to train the CNN.
#   4th: This corresponds to the test set on which we want to evaluate the performance of our CNN.
#   5th: This corresponds to the number of images in our test set.
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000/32,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000/32)



# Make new predictions
import numpy as np
from keras.preprocessing import image
# Loading the image
# Arguments:
#   1st: Image we want to test
#   2nd: Size of the image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))

# We need to add a new dimension because the input shape in the input layer of our CNN has three dimensions,
#   64 x 64 x 3. The 3 is because it is a colored image RGB
# The .img_to_array function will allow us to add that 3rd dimension.
# Arguments:
#   The Test Image
test_image = image.img_to_array(test_image)

# We need to add another dimension. 4 Dimensions
# We need to do this because the predict method requires it.
# This new dimension corresponds to the batch. Functions like the predict functions cannot accept a 
#   single input by itself, like the image we have here. It only accepts inputs in a batch. Even if the 
#   batch contains 1 input, the input must be in the batch.
# So, here we will have 1 batch of 1 input.
# Arguments:
#   1st: test image we want to expand
#   2nd: This is the axis. Axis is to specify the position of the index of the dimension that we are adding
#       We need to add it in the first position, because that's what the predict method expects.
#       So, we will specify axis = 0.
test_image = np.expand_dims(test_image, axis = 0)
# Now we use the predict function
result = classifier.predict(test_image)
# This will tell us exactky the mapping between the strings Cats and Dogs and their associated numeric
#   values, zero and one.
training_set.class_indices

# Now, we show the prediction as a string.
if result[0][0] == 1:
    prediction = 'Dog'
else:
    prediction = 'Cat'