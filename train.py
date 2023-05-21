import tensorflow as tf
import os
import numpy as np
import ecoset
from sklearn.utils.class_weight import compute_sample_weight


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
    nSub = 0
    for folder in basicClasses:
        files = os.listdir(os.path.join(directory, folder))
        if os.path.isdir(os.path.join(directory, folder, files[0])):
            nSub = len(files) + 1  # +1 for non-bird
            break

    imgPaths = []
    labels = []
    basicCounts = np.array([])
    subCounts = np.array([0])
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
                tf.TensorSpec(shape=(), dtype=tf.string),  # type: ignore
                tf.TensorSpec(shape=(), dtype=tf.int32),  # type: ignore
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

    # Turn lists into tensors
    imgPaths = tf.convert_to_tensor(imgPaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    ds = (
        tf.data.Dataset.from_tensor_slices((imgPaths, labels))
        .shuffle(len(imgPaths))
        .map(_parse_image)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Print dataset info
    print(f"Found {len(imgPaths)} images belonging to {len(classes)} classes.")

    return ds, weights


def create_twohot_dataset(directory, size=224, channel_first=False, batch_size=32):
    """
    Return a dataset given a nested directory structure. If the directory
    contains images, those images will be assigned to that class as one-hot. If
    the directory contains subdirectories, those subdirectories will be treated
    as two-hot where the first class is the parent directory and the second
    class is the subdirectory.
    """
    # List folders in directory
    basicClasses = os.listdir(directory)
    basicClasses.sort()
    nBasic = len(basicClasses)

    # Find the basic class that has subdirectories and count subclasses
    twoHots = {}
    for i, folder in enumerate(basicClasses):
        files = os.listdir(os.path.join(directory, folder))
        if os.path.isdir(os.path.join(directory, folder, files[0])):
            twoHots[i] = len(files)

    # Get total number of classes
    nSub = sum(twoHots.values())
    nClasses = nBasic + nSub

    labels = []
    imgPaths = []
    uniqueLabels = []
    basicCounts = np.array([])
    subCounts = np.array([])
    subClassCount = 0
    for i, folder in enumerate(basicClasses):
        # List files in folder
        files = os.listdir(os.path.join(directory, folder))
        files.sort()

        # Create label
        label = [i]

        # Check if this directory has directories in it
        if os.path.isdir(os.path.join(directory, folder, files[0])):
            # List folders in this directory
            subClasses = os.listdir(os.path.join(directory, folder))
            subClasses.sort()

            for subDir in subClasses:
                # List files in this directory
                files = os.listdir(os.path.join(directory, folder, subDir))
                files.sort()

                # Copy label and add an extra label
                subLabel = label[:]
                subLabel += [nBasic + subClassCount]
                subClassCount += 1

                # Add to unique labels
                uniqueLabels.append(subLabel)

                # Add to labels
                labels += [subLabel] * len(files)

                # Add to subclass counts
                subCounts = np.append(subCounts, len(files))

                for file in files:
                    imgPaths.append(os.path.join(directory, folder, subDir, file))

            # Add to basic class counts
            basicCounts = np.append(basicCounts, np.sum(subCounts))

        else:
            # Add to unique labels
            uniqueLabels.append(label)

            # Add to labels
            labels += [label] * len(files)

            # Add to class counts
            basicCounts = np.append(basicCounts, len(files))
            for file in files:
                imgPaths.append(os.path.join(directory, folder, file))

    # Convert imgPaths and labels to tensors
    imgPaths = tf.constant(imgPaths)
    labels = tf.ragged.constant(labels)

    counts = np.append(basicCounts, subCounts)
    weights = np.sum(counts) / (len(counts) * counts)
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)

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

        # Make one hot label with the first element of y
        label = tf.one_hot(y[0], len(counts))

        # If y has a second element, turn that into a one hot and add it to y
        if len(y) > 1:
            label2 = tf.one_hot(y[1], len(counts))
            label = tf.add(label, label2)

        return x, label

    ds = (
        tf.data.Dataset.from_tensor_slices((imgPaths, labels))
        .shuffle(len(imgPaths))
        .map(_parse_image)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(
        f"Found {len(imgPaths)} images belonging to {nClasses} classes with {nSub} subclasses in {int(np.sum(subCounts))} images."
    )

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
    model,
    trainDs,
    valDs,
    class_weights,
    lr,
    callbacks=[],
    initial_train=False,
    batch_norm=False,
):
    """
    Take an AlexNet model and perform transfer learning on it to classify the
    ecoCUB dataset.
    """
    # Freeze all layers
    for layer in model.layers:
        layer.trainable = False

    # Get old weights in fc8
    oldWeights, oldBias = model.layers[-4].get_weights()

    # Delete the old bird node (index 25)
    newWeights = np.delete(oldWeights, 25, axis=-1)
    newBias = np.delete(oldBias, 25)

    # Add new nodes
    weightInit = tf.keras.initializers.TruncatedNormal(stddev=0.005)
    newWeights = np.concatenate(
        [newWeights, weightInit(shape=(1, 1, 4096, 200)).numpy()], axis=-1
    )
    newBias = np.concatenate([newBias, np.zeros(200)])

    # Get model output at fc dropout layer
    x = model.layers[-5].output

    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    # Add new classification layer
    x = tf.keras.layers.Conv2D(
        764,
        (1, 1),
        padding="same",
        activation=None,
        name="birdFC",
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
    )(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Softmax()(x)

    # Create new model
    model = tf.keras.Model(inputs=model.input, outputs=[x])

    # Change birdFC layer weights and bias
    model.get_layer("birdFC").set_weights([newWeights, newBias])

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


def train_twohot_model(
    model,
    trainDs,
    valDs,
    class_weights,
    lr,
    softmax=True,
    callbacks=[],
    batch_norm=False,
):
    origKernel, origBias = model.get_layer("fc8").get_weights()
    weightInit = tf.keras.initializers.TruncatedNormal(
        stddev=0.005
    )  # It might be better to sample from old weights
    newWeights = weightInit(shape=(1, 1, 4096, 200))
    newBias = tf.keras.initializers.zeros()(shape=(200,))

    # Concatenate
    newWeights = tf.concat([origKernel, newWeights], axis=3)
    newBias = tf.concat([origBias, newBias], axis=0)

    # Freeze all layers
    for layer in model.layers:
        layer.trainable = False

    # Get model output at fc dropout layer
    x = model.layers[-4].output

    # Add new classification layer
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(
        765,
        (1, 1),
        padding="same",
        activation=None,
        name="twoHotFC",
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
    )(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)

    if softmax:
        x = tf.keras.layers.Softmax()(x)
        loss = tf.keras.losses.CategoricalCrossentropy()
    else:
        x = tf.keras.layers.Activation("sigmoid")(x)
        loss = tf.keras.losses.BinaryCrossentropy()

    # Create new model
    model = tf.keras.Model(inputs=model.input, outputs=[x])

    # Plugin the old weights with new concatenated
    model.get_layer("twoHotFC").set_weights([newWeights, newBias])

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=0.1),
        loss=loss,
        metrics=["accuracy", "top_k_categorical_accuracy", BirdAccuracy()],
    )
    model.summary()

    # Turn class weights into dictionary
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    # Train model
    fit = model.fit(
        trainDs,
        epochs=10,
        validation_data=valDs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    return fit


class BirdAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="bird_accuracy", **kwargs):
        super(BirdAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Find the samples with two-hot
        trueSums = tf.reduce_sum(y_true, axis=1)
        birdIndices = tf.where(tf.equal(trueSums, 2))
        birdIndices = tf.squeeze(birdIndices)

        # Get the true and predicted labels for those samples
        y_true = tf.gather(y_true, birdIndices)
        y_pred = tf.gather(y_pred, birdIndices)

        # Only keep the last 200 classes
        y_true = y_true[-200:]
        y_pred = y_pred[-200:]

        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        correct = tf.cast(tf.equal(y_true, y_pred), tf.float32)
        self.total.assign_add(tf.reduce_sum(correct))
        self.count.assign_add(tf.cast(tf.size(correct), tf.float32))

    def result(self):
        return self.total / self.count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a model for the deep cats project."
    )
    parser.add_argument(
        "--script",
        type=str,
        help="type of model to train",
        choices=["ecoCubAmnesia", "twoHot"],
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
    parser.add_argument(
        "--dataDir",
        type=str,
        help="directory containing the training data",
        default="./images/",
    )

    args = parser.parse_args()

    seed = args.seed

    if args.script == "ecoCubAmnesia":
        augment = args.augment
        batchNorm = args.batchNorm
        dataDir = args.dataDir

        # Training seed
        tf.random.set_seed(seed)
        size = 256 if augment else 224

        # Create dataset
        trainDs, weights = create_flat_dataset(
            os.path.join(dataDir, "ecoCUB", "train"), size=size
        )
        valDs, _ = create_flat_dataset(
            os.path.join(dataDir, "ecoCUB", "val"), size=size, filter="CUB"
        )

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
            trainDs=trainDs,
            valDs=valDs,
            class_weights=weights,
            lr=0.001,
            callbacks=callbacks,
            batch_norm=batchNorm,
        )
    elif args.script == "twoHot":
        weightPath = f"./models/AlexNet/ecoset_training_seeds_01_to_10/training_seed_{seed:02}/model.ckpt_epoch89"
        model = ecoset.make_alex_net_v2(
            weights_path=weightPath,
            input_shape=(224, 224, 3),
        )

        trainDs, weights = create_twohot_dataset(
            os.path.join(args.dataDir, "train"),
            size=224,
            channel_first=False,
            batch_size=32,
        )
        valDs, _ = create_twohot_dataset(
            os.path.join(args.dataDir, "val"),
            size=224,
            channel_first=False,
            batch_size=32,
        )

        fit = train_twohot_model(
            model=model,
            trainDs=trainDs,
            valDs=valDs,
            lr=0.0001,
            class_weights=weights,
            batch_norm=args.batchNorm,
        )
