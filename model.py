import csv
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import matplotlib.image


# center,left,right,steering,throttle,brake,speed
class DataEntry:
    def __init__(self, row):
        self.center = row[0]
        self.left = row[1]
        self.right = row[2]
        self.steering = row[3]
        self.throttle = row[4]
        self.brake = row[5]
        self.speed = row[6]


def read_csv_info(input_path_tuples):
    rows = []
    for csv_file_path, directory_path in input_path_tuples:
        with open(csv_file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if row[0] == 'center':
                    continue
                data_entry = DataEntry(row)
                data_entry.center = os.path.join(directory_path, data_entry.center.strip())
                data_entry.left = os.path.join(directory_path, data_entry.left.strip())
                data_entry.right = os.path.join(directory_path, data_entry.right.strip())
                rows.append(data_entry)
    return rows


# def augmentData(x_data, y_data , is_train):
#     output_x, output_y = [], []
#     for img, measure in zip(x_data, y_data):
#         output_x.append(img)
#         output_y.append(measure)
#         output_x.append(cv2.flip(img, 1))
#         output_y.append(measure * -1.0)
#
#     return np.array(output_x), np.array(output_y)

def generator(samples, batch_size, is_training):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample.center
                center_image = cv2.imread(name)
                center_angle = float(batch_sample.steering)

                images.append(center_image)
                angles.append(center_angle)
                correction = 0.2
                left_image = cv2.imread(batch_sample.left)
                left_angle = center_angle + correction
                images.append(left_image)
                angles.append(left_angle)
                right_image = cv2.imread(batch_sample.right)
                right_angle = center_angle - correction
                images.append(right_image)
                angles.append(right_angle)
                # If it is in training mode flip the image
                if (is_training):
                    images.append(cv2.flip(center_image, 1))
                    angles.append(center_angle * -1.0)

            # trim image to only see section with road
            x_data = np.array(images)
            y_data = np.array(angles)
            # shapes = set([image.shape for image in images])
            yield sklearn.utils.shuffle(x_data, y_data)


def show_random_flip_example():
    data = get_data_entry_list()
    sample = random.choice(data)
    img = cv2.imread(sample.center)
    matplotlib.image.imsave("images/image_original1.png", img)
    matplotlib.image.imsave("images/image_flip1.png", cv2.flip(img, 1))


def get_data_entry_list():
    input_path_tuples = [
        ("data/driving_log.csv", 'data'),
        ("collected_data/driving_log.csv", ''),
        ("curve_saver/driving_log.csv", ''),
        ("reverse_lane/driving_log.csv", ''),
        ("speicial_edge/driving_log.csv", ''),
        ("right_turn/driving_log.csv", ''),
        ("low_resolution/driving_log.csv", '')
    ]
    data_entry_list = read_csv_info(input_path_tuples)
    return data_entry_list


def main():

    data_entry_list = get_data_entry_list()

    # data_entry_list = data_entry_list[:1000]
    data_entry_list = sklearn.utils.shuffle(data_entry_list)
    train_samples, validation_samples = train_test_split(data_entry_list, test_size=0.2)

    train_generator = generator(train_samples, batch_size=128, is_training=True)
    validation_generator = generator(validation_samples, batch_size=128, is_training=True)

    # define model
    input_shape = (160, 320, 3)
    model = Sequential()
    model.add(Lambda(lambda x: (x -128) / 128.0, input_shape=input_shape))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', lr=0.0001)

    # train model
    history_object = model.fit_generator(train_generator, samples_per_epoch=
    len(train_samples) , validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)
    model.save('model.h5')

    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('model_error_loss.png')
    pass


if __name__ == '__main__':
    # show_random_flip_example()
    main()
