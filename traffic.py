from PIL import Image
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

#EPOCHS = 10
#IMG_WIDTH = 30
#IMG_HEIGHT = 30
#NUM_CATEGORIES = 43
#TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=0.4
    )
    
    # Prepare data for training
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    x_train = x_train.reshape(
        x_train.shape[0], x_train.shape[1], x_train.shape[2], 3
    )
    #print(x_train.shape[0])
    #print(x_train.shape[1])
    #print(x_train.shape[2])
    
    x_test = x_test.reshape(
        x_test.shape[0], x_test.shape[1], x_test.shape[2], 3
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=10)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    `data_dir` - directory named after each category, numbered
    0 through # of categories (43) - 1
    
    inside each category directory is some
    number of image files.

    """
    #data_dir="/Users/bibianamailyn/Documents/Machine_learning/Python_code/src5/traffic/gtsrb"
    
    # List to store loaded images
    images = []
    labels = []

    # Iterate through all files in the directory
    for category_dir in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, category_dir)):
            # Get the category label from the directory name
            category_label = int(category_dir)
            #print(category_dir)
            # Iterate through all files in the category directory
            for filename in os.listdir(os.path.join(data_dir, category_dir)):
                if filename.endswith(('.ppm')):  # Add other formats as needed
                    # Construct the full file path
                    file_path = os.path.join(data_dir, category_dir, filename)
                    #print(file_path)
                    # Read the image using Pillow
                    img = Image.open(file_path)
                    
                    # Resize the image to the specified dimensions
                    img = img.resize((30, 30))

                    # Print the resized dimensions
                    #print("Resized dimensions:", img.size)
                    
                    # Convert the image to a NumPy array
                    img_array = np.array(img)

                    # Append the image and label to the lists
                    # Convert the list of images to a 4D NumPy array
                    images.append(img_array) 
                    labels.append(category_label)
    
    #`images` = list of all images in the data directory, where each image is formatted as a numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3
    images = np.stack(images, axis=0)       

    return images, labels #returns tuple `(images, labels)`



def get_model():
    """
    input_shape of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    
    output layer: `NUM_CATEGORIES` units, = 43
    """
    
    
    # Create a convolutional neural network
   
    model = tf.keras.models.Sequential([

       # Convolutional layer. Learn 32 filters using a 3x3 kernel
       tf.keras.layers.Conv2D(
           32, (3, 3), activation="relu", input_shape=(30, 30, 3)
       ),

       # Max-pooling layer, using 2x2 pool size
       tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

       # Flatten units
       tf.keras.layers.Flatten(),

       # Add a hidden layer with dropout of 0.5
       tf.keras.layers.Dense(128, activation="relu"),
       tf.keras.layers.Dropout(0.5),

       # Add an output layer with output units for all 43 categories
       tf.keras.layers.Dense(43, activation="softmax") #softmax to get probabilty for the correctness of the digit
   ])
    
    # Train neural network
    model.compile(
       optimizer="adam",
       loss="categorical_crossentropy",
       metrics=["accuracy"]
   )
    
    return model #returns a compiled convolutional neural network model


if __name__ == "__main__":
    main()

