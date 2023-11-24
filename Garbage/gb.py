import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

batch_size = 256
num_classes = 10
epochs = 100

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = [img_rows, img_cols, 1]

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

num_zeros = 10000
garbage_train_x = np.random.rand(num_zeros, 28, 28, 1)
zero_val = np.exp(0)/(np.exp(0)*10)
garbage_train_y = np.full([num_zeros, 10], zero_val)

x_train = np.concatenate([garbage_train_x, x_train], axis=0)
y_train = np.concatenate([garbage_train_y, y_train], axis=0)

x_train, y_train = unison_shuffled_copies(x_train, y_train)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

model = tf.keras.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])