from numpy import average
import tensorflow as tf
from tensorflow.keras import layers


def make_output_model(model, average_pooling=True, flatten=True):
    """Return a model that has outputs at every convolution and dense layer."""
    outputs = []
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            dataFormat = layer.data_format
            if average_pooling:
                layer = layers.GlobalAvgPool2D(data_format=dataFormat)(
                    layer.output
                )

            if flatten:
                if average_pooling:
                    layer = layers.Flatten(data_format=dataFormat)(layer)
                else:
                    layer = layers.Flatten(data_format=dataFormat)(
                        layer.output
                    )

            if average_pooling or flatten:
                outputs.append(layer)
            else:
                outputs.append(layer.output)
        elif isinstance(layer, layers.Dense):
            outputs.append(layer.output)

    return tf.keras.Model(model.input, outputs)
