import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# loading data
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# preprocessing data

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
#
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=3)

# model.save("handwritten.model")

model = tf.keras.models.load_model('handwritten.model')

# ls, acc = model.evaluate(x_test, y_test)

# print(ls)
# print(acc)

img_num = 1
while os.path.isfile(f"digits/digit{img_num}.png"):
    try:
        img= cv2.imread(f"digits/digit{img_num}.png")[:,:,0]
        img = np.invert(np.array([img]))
        pred = model.predict(img)
        print(f"This digit is a {np.argmax(pred)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
    except:
        print("Error!!!")
    finally:
        img_num += 1






