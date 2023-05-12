import tensorflow as tf
import os
import numpy as np
import ecoset


def create_nested_dataset(directory, size=224, channel_first=False, batch_size=32):
    """
    Create tf.Data.Dataset objects from the train and val directories, assumes
    that we have one directory that has subdirectories so we will have multi-
    class classification. The dataset will be resized to size x size. The
    dataset will be batched and shuffled.
    """
    # List folders in trainDir
    basicClasses = os.listdir(directory)
    basicClasses.sort()
    nBasic = len(basicClasses)

    # Find the basic class that has subdirectories and count the number
    for folder in basicClasses:
        files = os.listdir(os.path.join(directory, folder))
        if os.path.isdir(os.path.join(directory, folder, files[0])):
            nSub = len(files) + 1  # +1 for non-bird
            break

    imgPaths = []
    labels = []
    basicCounts = np.array([])
    for i, folder in enumerate(basicClasses):
        basicCount = 0
        # List files in folder
        files = os.listdir(os.path.join(directory, folder))
        files.sort()

        # Check if this directory has directories in it
        if os.path.isdir(
            os.path.join(directory, folder, files[0])
        ):  # This is probably birds
            # List folders in this directory
            subClasses = os.listdir(os.path.join(directory, folder))
            subClasses.sort()
            subCounts = np.array([])
            for j, subDir in enumerate(subClasses):
                # List files in this directory
                files = os.listdir(os.path.join(directory, folder, subDir))
                files.sort()

                subCount = 0
                for file in files:
                    imgPaths.append(os.path.join(directory, folder, subDir, file))
                    labels.append((i, j))
                    basicCount += 1
                    subCount += 1

                subCounts = np.append(subCounts, subCount)
        else:  # Not birds
            for file in files:
                imgPaths.append(os.path.join(directory, folder, file))
                labels.append((i, nSub))
                basicCount += 1

        basicCounts = np.append(basicCounts, basicCount)

    # Add non-bird count
    subCounts = np.append(subCounts, np.sum(basicCounts) - np.sum(subCounts))

    def _parse_image(x, y):
        # Decode image
        x = tf.io.read_file(x)
        x = tf.io.decode_image(x, channels=3)

        # Cast to float
        x = tf.cast(x, tf.float32)

        # Resize
        x = tf.keras.preprocessing.image.smart_resize(x, (size, size))

        # Center features
        x = 2 * (x / 255 - 0.5)

        # Transpose to channel first format
        if channel_first:
            x = tf.transpose(x, (2, 0, 1))

        # One-hot encode labels
        y = (tf.one_hot(y[0], nBasic), tf.one_hot(y[1], nSub))

        return x, y

    ds = (
        tf.data.Dataset.from_generator(
            lambda: zip(imgPaths, labels),
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(2,), dtype=tf.int32),
            ),
        )
        .shuffle(len(imgPaths))
        .map(_parse_image)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Calculate category weights
    basicWeights = np.sum(basicCounts) / (nBasic * basicCounts)
    subWeights = np.sum(subCounts) / (nSub * subCounts)

    # Convert weights to tensors
    basicWeights = tf.convert_to_tensor(basicWeights, dtype=tf.float32)
    subWeights = tf.convert_to_tensor(subWeights, dtype=tf.float32)

    # Print class counts
    print(
        f"Found {len(imgPaths)} images belonging to {nBasic} basic classes and {nSub} subordinate classes."
    )

    return ds, basicWeights, subWeights


def create_flat_dataset(
    directory, size=224, channel_first=False, batch_size=32, filter=None
):
    """
    Return a dataset and class weights from directory where all images are
    scaled to size x size.
    """
    # List folders in directory
    classes = os.listdir(directory)
    classes.sort()

    # Create lists
    imgPaths = []
    labels = []
    classCounts = np.array([])
    for i, folder in enumerate(classes):
        if (filter is not None) and (filter not in folder):
            classCounts = np.append(classCounts, 0)
            continue

        # List files in folder
        files = os.listdir(os.path.join(directory, folder))
        files.sort()

        imgCount = 0
        for file in files:
            imgPaths.append(os.path.join(directory, folder, file))
            labels.append(i)
            imgCount += 1

        classCounts = np.append(classCounts, imgCount)

    # Calculate class weights
    if filter is None:
        weights = np.sum(classCounts) / (len(classes) * classCounts)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    else:
        weights = None

    def _parse_image(x, y):
        # Decode image
        x = tf.io.read_file(x)
        x = tf.io.decode_image(x, channels=3)

        # Cast to float
        x = tf.cast(x, tf.float32)

        # Resize
        x = tf.keras.preprocessing.image.smart_resize(x, (size, size))

        # Center features
        x = 2 * (x / 255 - 0.5)

        # Transpose to channel first format
        if channel_first:
            x = tf.transpose(x, (2, 0, 1))

        # One-hot encode labels
        y = tf.one_hot(y, len(classes))

        return x, y

    ds = (
        tf.data.Dataset.from_generator(
            lambda: zip(imgPaths, labels),
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )
        .shuffle(len(imgPaths))
        .map(_parse_image)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Print dataset info
    print(f"Found {len(imgPaths)} images belonging to {len(classes)} classes.")

    return ds, weights


def make_expert_model(
    model,
    layer_idx,
    num_layers,
    n_thaw,
    basicWeights=None,
    subWeights=None,
    loss_weights=[1, 1],
):
    """
    Return a copy of the model with an additional expert branch added at
    layer_idx. The expert branch will be num_layers layers deep. All old layers
    behind layer_idx - n_thaw will be frozen.
    """
    # Turn negative indices positive
    if layer_idx < 0:
        layer_idx = len(model.layers) + layer_idx

    # Freeze layers
    for layer in model.layers[: layer_idx - n_thaw]:
        print(f"Freezing layer {layer.name}")
        layer.trainable = False

    # Get output
    outputs = model.outputs

    # Get the layer to add to
    layer = model.layers[layer_idx]
    print(f"Adding branch to layer {layer.name}")
    x = layer.output

    for _ in range(num_layers):
        x = tf.keras.layers.Conv2D(1024, 3, activation="relu", padding="same")(x)

    # Add expert branch
    x = tf.keras.layers.Conv2D(201, 5, activation=None, padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Softmax()(x)

    # Create new model
    model = tf.keras.Model(inputs=model.input, outputs=outputs + [x])

    if basicWeights is None or subWeights is None:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=[
                tf.keras.losses.CategoricalCrossentropy(),
                tf.keras.losses.CategoricalCrossentropy(),
            ],
            metrics=["accuracy"],
            loss_weights=loss_weights,
        )

    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=[
                weighted_cce(basicWeights),
                weighted_cce(subWeights),
            ],
            metrics=["accuracy"],
            loss_weights=loss_weights,
        )

    return model


class weighted_cce(tf.keras.losses.Loss):
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.weights = weights

    def call(self, y_true, y_pred, sample_weight=None):
        weights = tf.gather(self.weights, tf.argmax(y_true, axis=-1))

        return tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred, weights)


def train_ecocub_model(
    model, class_weights, lr, callbacks=[], initial_train=False, batch_norm=False
):
    """
    Take an AlexNet model and perform transfer learning on it to classify the
    ecoCUB dataset.
    """
    # Freeze all layers
    for layer in model.layers:
        layer.trainable = False

    # Get model output at fc dropout layer
    x = model.layers[-5].output

    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    # Add new classification layer
    weightInit = tf.keras.initializers.TruncatedNormal(stddev=0.005)
    x = tf.keras.layers.Conv2D(
        764,
        (1, 1),
        padding="same",
        activation=None,
        name="birdFC",
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        kernel_initializer=weightInit,
        bias_initializer=tf.keras.initializers.zeros(),
    )(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Softmax()(x)

    # Create new model
    model = tf.keras.Model(inputs=model.input, outputs=[x])

    # Turn class weights into dictionary
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    if initial_train:
        print("Training one epoch to line up clasification layer")
        # Train one epoch to line up the new classification layer
        model.compile(
            optimizer=tf.keras.optimizers.Adam(epsilon=0.1),
            loss=tf.keras.losses.CategoricalCrossentropy(),
        )

        model.summary()

        # Fit one epoch
        model.fit(
            trainDs,
            epochs=1,
            validation_data=valDs,
            class_weight=class_weights,
        )

    # Unfreeze penultimate layer and add regularizer
    model.layers[-6].trainable = True
    model.layers[-6].kernel_regularizer = tf.keras.regularizers.l2(0.0005)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=0.1),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy", "top_k_categorical_accuracy"],
    )

    model.summary()

    # Train model
    fit = model.fit(
        trainDs,
        epochs=10,
        validation_data=valDs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    return fit


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a model for the deep cats project."
    )
    parser.add_argument(
        "--script",
        type=str,
        help="type of model to train",
        choices=["birder"]
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed to use for training",
        required=True,
    )
    parser.add_argument(
        "--augment",
        type=bool,
        help="whether to use data augmentation",
        default=False,
    )
    parser.add_argument(
        "--batchNorm",
        type=bool,
        help="whether to use batch normalization",
        default=False,
    )

    args = parser.parse_args()

    seed = args.seed

    if args.script == "birder":
        augment = args.augment
        batchNorm = args.batchNorm

        # Training seed
        tf.random.set_seed(seed)
        size = 256 if augment else 224

        # Create dataset
        trainDs, weights = create_flat_dataset("./images/ecoCUB/train", size=size)
        valDs, _ = create_flat_dataset("./images/ecoCUB/val", size=size, filter="CUB")

        weightPath = f"./models/AlexNet/ecoset_training_seeds_01_to_10/training_seed_{seed:02}/model.ckpt_epoch89"
        model = ecoset.make_alex_net_v2(
            weights_path=weightPath,
            softmax=True,
            augment=augment,
            input_shape=(size, size, 3),
        )

        # Make callbacks
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f"./models/deepCats/AlexNet/seed{seed:02}/epoch{{epoch:02d}}-val_loss{{val_loss:.2f}}.hdf5",
            monitor="val_loss",
            save_freq="epoch",
        )
        csvLogger = tf.keras.callbacks.CSVLogger(
            f"./models/deepCats/AlexNet/seed{seed:02}/training.csv", append=True
        )

        def exp_schedule(epoch):
            lr = 0.001
            return lr * tf.math.pow(0.5, epoch)

        schedule = tf.keras.callbacks.LearningRateScheduler(exp_schedule, verbose=1)
        callbacks = [checkpoint, csvLogger, schedule]

        # Train model
        fit = train_ecocub_model(
            model=model,
            class_weights=weights,
            lr=0.001,
            callbacks=callbacks,
            batch_norm=batchNorm,
        )
        