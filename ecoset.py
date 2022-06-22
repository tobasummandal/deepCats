import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


def make_alex_net_v2(
    input_shape=(224, 224, 3), output_shape=565, weights_path=None
):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(
        64,
        (11, 11),
        strides=(4, 4),
        padding="valid",
        activation="relu",
        name="conv1",
    )(inputs)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), name="pool1")(x)

    x = layers.Conv2D(
        192,
        (5, 5),
        padding="same",
        activation="relu",
        name="conv2",
    )(x)
    x = layers.MaxPool2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="pool2",
    )(x)

    x = layers.Conv2D(
        384,
        (3, 3),
        padding="same",
        activation="relu",
        name="conv3",
    )(x)
    x = layers.Conv2D(
        384,
        (3, 3),
        padding="same",
        activation="relu",
        name="conv4",
    )(x)
    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu",
        name="conv5",
    )(x)
    x = layers.MaxPool2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="pool5",
    )(x)

    x = layers.Conv2D(
        4096,
        (5, 5),
        padding="same",
        activation="relu",
        name="fc6",
    )(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(
        4096,
        (1, 1),
        padding="same",
        activation="relu",
        name="fc7",
    )(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(
        output_shape,
        (1, 1),
        padding="same",
        activation="relu",
        name="fc8",
    )(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)

    model = tf.keras.Model(inputs, x)

    if weights_path is not None:
        load_alexnet_weights(model, weights_path)

    return model


def make_vNet(input_shape=(128, 128, 3), output_shape=565, weights_path=None):
    inputs = tf.keras.Input(shape=input_shape)

    # Layer 1 - V1
    x = layers.Conv2D(128, (7, 7), name="conv1", padding="same")(inputs)
    x = tfa.layers.GroupNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Layer 2 - V2
    x = layers.Conv2D(128, (7, 7), name="conv2", padding="same")(x)
    x = tfa.layers.GroupNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Layer 3 - V3
    x = layers.MaxPool2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Conv2D(256, (5, 5), name="conv3", padding="same")(x)
    x = tfa.layers.GroupNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Layer 4 - hV4
    x = layers.MaxPool2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Conv2D(256, (5, 5), name="conv4", padding="same")(x)
    x = tfa.layers.GroupNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Layer 5 - LO
    x = layers.Conv2D(512, (3, 3), name="conv5", padding="same")(x)
    x = tfa.layers.GroupNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Layer 6 - TO
    x = layers.Conv2D(512, (3, 3), name="conv6", padding="same")(x)
    x = tfa.layers.GroupNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Layer 7 - pFUS
    x = layers.MaxPool2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Conv2D(1024, (3, 3), name="conv7", padding="same")(x)
    x = tfa.layers.GroupNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Layer 8 - mFUS
    x = layers.MaxPool2D(pool_size=(2, 2), name="pool4")(x)
    x = layers.Conv2D(1024, (3, 3), name="conv8", padding="same")(x)
    x = tfa.layers.GroupNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Layer 9
    x = layers.MaxPool2D(pool_size=(2, 2), name="pool5")(x)
    x = layers.Conv2D(2048, (1, 1), name="conv9", padding="same")(x)
    x = tfa.layers.GroupNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Layer 10
    x = layers.MaxPool2D(pool_size=(2, 2), name="pool6")(x)
    x = layers.Conv2D(2048, (1, 1), name="conv10", padding="same")(x)
    x = tfa.layers.GroupNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Readout
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(output_shape, name="readout")(x)

    model = tf.keras.Model(inputs, x)

    if weights_path is not None:
        load_alexnet_weights(model, weights_path)

    return model


def load_alexnet_weights(model, weights_path):
    """Loads the weights into a model from a file in place."""
    reader = tf.train.load_checkpoint(weights_path)
    dtypes = reader.get_variable_to_dtype_map()

    # Loop through layers of the model
    for layer in model.layers:
        # Check if the layer is a conv layer
        if isinstance(layer, tf.keras.layers.Conv2D):
            weightKeys = [
                key
                for key in dtypes.keys()
                if layer.name in key and "Momentum" not in key
            ]
            weightKeys.sort()
            # Get checkpoint weights
            weights = reader.get_tensor(weightKeys[1])
            bias = reader.get_tensor(weightKeys[0])

            # Set weights and biases
            layer.set_weights([weights, bias])

    print(f"Weights from {weights_path} loaded successfully.")
