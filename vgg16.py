from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
import numpy as np
from keras import backend

def load_data(name='cifar10', file='data_batch_1'):
    import _pickle
    f = open('datasets/' + name + '/' + file, 'rb')
    dict = _pickle.load(f, encoding='latin1')
    images = dict['data']
    backend.set_image_data_format('channels_first')
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict['labels']
    X = np.array(images)  # (10000, 3072)
    y = np.array(labels)  # (10000,)
    return X, y

def split_data(X, y, test_size=0.33, seed=0):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=seed)

def vgg16_model(height, width, channels=3, num_classes=None):
    model = Sequential([
        Conv2D(64, (3, 3), input_shape=(channels, width, height), padding='same',
               activation='relu'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same', ),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(256, (3, 3), activation='relu', padding='same', ),
        Conv2D(256, (3, 3), activation='relu', padding='same', ),
        Conv2D(256, (3, 3), activation='relu', padding='same', ),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same', ),
        Conv2D(512, (3, 3), activation='relu', padding='same', ),
        Conv2D(512, (3, 3), activation='relu', padding='same', ),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same', ),
        Conv2D(512, (3, 3), activation='relu', padding='same', ),
        Conv2D(512, (3, 3), activation='relu', padding='same', ),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        # Flatten(),
        # Dense(4096, activation='relu'),
        # Dense(4096, activation='relu'),
        # Dense(1000, activation='softmax')
    ])

    print(model.summary())

    for layer in model.layers[:15]:
        layer.trainable = False

    model.load_weights('pretrained_weights/vgg16_weights_notop.h5')

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':

    img_rows, img_cols = 32, 32 # Resolution of inputs
    channel = 3
    num_classes = 10
    batch_size = 16
    nb_epoch = 10

    from sklearn.metrics import log_loss
    X, y = load_data('CIFAR10')
    X_train, X_valid, Y_train, Y_valid = split_data(X, y)

    model = vgg16_model(img_rows, img_cols, channel, num_classes)

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              )

    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    score = log_loss(Y_valid, predictions_valid)