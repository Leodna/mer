from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


def vgg16_svm():
    base_model = VGG16(
        weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3)
    )
    inputs = Input((224, 224, 3))
    prep_input = preprocess_input(inputs)
    base_outputs = base_model(prep_input)

    return Model(inputs, base_outputs, name="feature_extractor")


def vgg16(trainable=False):
    base_model = VGG16(
        weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3)
    )

    if trainable:
        for layer in base_model.layers:
            layer.trainable = True

    inputs = Input((224, 224, 3))
    prep_input = preprocess_input(inputs)
    base_outputs = base_model(prep_input)

    x = layers.Flatten()(base_outputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)

    outputs = layers.Dense(2, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mae"])

    model.summary()

    return model


def modelo1():
    inputs = Input(shape=(513, 1671, 1))

    x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(inputs)

    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(x)

    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(x)

    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same")(x)

    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)

    x = layers.Dense(128, activation="relu")(x)

    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation="relu")(x)

    x = layers.Dense(2, activation="linear")(x)

    model = Model(inputs=inputs, outputs=x)

    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mae"])

    model.summary()

    return model


# def modelo1():
#     inputs = Input(shape=(513, 1671, 1))

#     # Primera capa
#     x_conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(
#         inputs
#     )
#     x_pooling1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x_conv1)
#     x_batch1 = layers.BatchNormalization()(x_pooling1)

#     # Segunda capa
#     x_conv2 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(
#         x_batch1
#     )
#     x_pooling2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x_conv2)
#     x_batch2 = layers.BatchNormalization()(x_pooling2)

#     # Tercer capa
#     x_conv3 = layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(
#         x_batch2
#     )
#     x_pooling3 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x_conv3)
#     x_batch3 = layers.BatchNormalization()(x_pooling3)

#     # Cuarta capa
#     x_conv4 = layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same")(
#         x_batch3
#     )
#     x_pooling4 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x_conv4)
#     x_batch4 = layers.BatchNormalization()(x_pooling4)

#     # Capa flatten
#     x_flatten = layers.Flatten()(x_batch4)

#     x_fc1 = layers.Dense(128, activation="relu")(x_flatten)
#     x_drop = layers.Dropout(0.5)(x_fc1)
#     x_fc2 = layers.Dense(64, activation="relu")(x_drop)
#     x_out = layers.Dense(2, activation="linear")

#     model = Model(inputs=inputs, outputs=x_out)
#     model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mae"])
#     model.summary()

#     return model
