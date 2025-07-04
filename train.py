import tensorflow as tf
import os
import numpy as np
import ecoset
from sklearn.utils.class_weight import compute_sample_weight
import glob


def create_nested_dataset(
    directory, size=224, channel_first=False, batch_size=32, not_birds=True
):
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
            nSub = len(files) + 1 if not_birds else len(files)  # +1 for non-bird
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

    # Convert imgPaths and labels to tensors
    imgPaths = tf.constant(imgPaths)
    labels = tf.ragged.constant(labels)

    # Add non-bird count
    subCounts = np.append(subCounts, np.sum(basicCounts) - np.sum(subCounts))

    @tf.function
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
        basicLabel = tf.one_hot(y[0], nBasic)

        if (not not_birds) and tf.equal(y[1], 200):
            subLabel = tf.zeros(nSub)
        else:
            subLabel = tf.one_hot(y[1], nSub)

        return x, (basicLabel, subLabel)

    ds = (
        tf.data.Dataset.from_tensor_slices((imgPaths, labels))
        .shuffle(len(imgPaths))
        .map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)
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
        # Check if this folder has folders in it
        if os.path.isdir(os.path.join(directory, folder, files[0])):
            # List folders in this directory
            subClasses = os.listdir(os.path.join(directory, folder))
            subClasses.sort()

            # Count number of images in this folder
            imgCount = 0
            for subDir in subClasses:
                # List files in this directory
                files = os.listdir(os.path.join(directory, folder, subDir))
                files.sort()

                files = [
                    os.path.join(directory, folder, subDir, file) for file in files
                ]
                imgPaths += files
                labels += [i] * len(files)

                imgCount += len(files)

            classCounts = np.append(classCounts, imgCount)
        else:
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


def create_twohot_dataset(
    directory,
    sub_cat,
    size=224,
    channel_first=False,
    batch_size=32,
    softmax_labels=False,
):
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

    # Find the index of the subcategory and the number of subordinate categories it has
    twoHots = {}
    twoHots[basicClasses.index(sub_cat)] = len(
        os.listdir(os.path.join(directory, sub_cat))
    )

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

        # Check if this is the sub directory
        if folder == sub_cat:
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
        # Check if this directory has directories in it
        elif os.path.isdir(os.path.join(directory, folder, files[0])):
            # Add to unique labels
            uniqueLabels.append(label)

            # List folders in this directory
            subClasses = os.listdir(os.path.join(directory, folder))
            subClasses.sort()

            subImgCounts = 0
            for subDir in subClasses:
                # List files in this directory
                files = os.listdir(os.path.join(directory, folder, subDir))
                files.sort()

                # Add to labels
                labels += [label] * len(files)

                for file in files:
                    imgPaths.append(os.path.join(directory, folder, subDir, file))

                subImgCounts += len(files)

            # Add to basic class counts
            basicCounts = np.append(basicCounts, subImgCounts)
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
        x = tf.divide(x, 255.0)
        x = tf.subtract(x, 0.5)
        x = tf.multiply(x, 2.0)

        # Transpose to channel first format
        if channel_first:
            x = tf.transpose(x, (2, 0, 1))

        # Make one hot label with the first element of y
        label = tf.one_hot(y[0], len(counts))

        # If y has a second element, turn that into a one hot and add it to y
        if tf.size(y) > 1:
            label2 = tf.one_hot(y[1], len(counts))
            label = tf.add(label, label2)
            if softmax_labels:
                label = label / 2.0

        return x, label

    ds = (
        tf.data.Dataset.from_tensor_slices((imgPaths, labels))
        .shuffle(len(imgPaths))
        .map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(
        f"Found {len(imgPaths)} images belonging to {nClasses} classes with {nSub} subclasses in {int(np.sum(subCounts))} images."
    )

    return ds, weights, twoHots


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
    epochs,
    callbacks=[],
    initial_train=False,
    l2_reg=True,
    batch_norm=False,
    reuse_weights=True,
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
        kernel_regularizer=tf.keras.regularizers.l2(0.0005) if l2_reg else None,
        kernel_initializer=weightInit,
        bias_initializer=tf.keras.initializers.Zeros(),
    )(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Softmax()(x)

    # Create new model
    model = tf.keras.Model(inputs=model.input, outputs=[x])

    if reuse_weights:
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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=0.1),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            "accuracy",
            "top_k_categorical_accuracy",
            OneHotBirdAccuracy(top_k=1, name="bird_top1"),
            OneHotBirdAccuracy(top_k=5, name="bird_top5"),
        ],
    )

    model.summary()

    # Train model
    fit = model.fit(
        trainDs,
        epochs=epochs,
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
    epochs,
    two_hots,
    sub_layers=0,
    thaw_layers=["fc7", "fc8", "subFC"],
    softmax=True,
    l2_reg=True,
    callbacks=[],
    batch_norm=False,
    reuse_weights=True,
):
    _, subNodes = list(two_hots.items())[0]
    # Get the output of the previous classification layer
    basicOutput = model.layers[-3].output

    # Get model output at fc dropout layer
    x = model.layers[-4].output

    # Add new classification layer
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    weightInit = tf.keras.initializers.TruncatedNormal(stddev=0.005)

    if sub_layers > 0:
        # Add extra layers
        for i in range(sub_layers):
            x = tf.keras.layers.Conv2D(
                4096,
                (1, 1),
                padding="same",
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.0005) if l2_reg else None,
                name=f"sub{i}",
            )(x)
            # Add layer to unfreeze
            thaw_layers.append(f"sub{i}")

    x = tf.keras.layers.Conv2D(
        subNodes,
        (1, 1),
        padding="same",
        activation=None,
        name="subFC",
        kernel_initializer=weightInit,
        kernel_regularizer=tf.keras.regularizers.l2(0.0005) if l2_reg else None,
    )(x)
    x = tf.keras.layers.Concatenate()([basicOutput, x])
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

    # Reinitialize fc8 weights if needed
    if not reuse_weights:
        # Remake initializer for weights to avoid identical values
        weightInit = tf.keras.initializers.TruncatedNormal(stddev=0.005)
        oldWeights, oldBias = model.get_layer("fc8").get_weights()
        newWeights = weightInit(tf.shape(oldWeights))
        newBias = tf.keras.initializers.Zeros()(tf.shape(oldBias))
        model.get_layer("fc8").set_weights([newWeights, newBias])

    # Freeze all layers
    for layer in model.layers:
        layer.trainable = False

    for layer in thaw_layers:
        model.get_layer(layer).trainable = True

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr, epsilon=0.1, weight_decay=0.0005
        ),
        loss=loss,
        metrics=[
            "accuracy" if softmax else tf.keras.metrics.BinaryAccuracy(),
            "top_k_categorical_accuracy",
            TwoHotSubAccuracy(sub_nodes=subNodes, top_k=1, name="sub_top1"),
            TwoHotSubAccuracy(sub_nodes=subNodes, top_k=5, name="sub_top5"),
        ],
    )
    model.summary()

    # Turn class weights into dictionary
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    # Train model
    fit = model.fit(
        trainDs,
        epochs=epochs,
        validation_data=valDs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    return fit


def train_branch_model(
    model,
    trainDs,
    valDs,
    basic_weights,
    sub_weights,
    lr,
    epochs,
    branch_layer="fc7",
    loss_weights=[1, 1],
    thaw_layers=["fc7", "fc8", "birdFC"],
    l2_reg=True,
    callbacks=[],
    non_bird_node=True,
):
    """
    Take an Alexmodel and modify it to have a branch for basic and subordinate
    classification. The branch is added after the layer specified by layer_idx.
    """
    # Get output
    outputs = model.outputs

    # Get the layer to add to
    layer = model.get_layer(branch_layer)
    print(f"Adding branch to layer {layer.name}")
    x = layer.output

    # Add expert branch
    subNodes = 201 if non_bird_node else 200
    x = tf.keras.layers.Conv2D(
        subNodes,
        5,
        activation=None,
        padding="same",
        name="birdFC",
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.005),
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_regularizer=tf.keras.regularizers.l2(0.0005) if l2_reg else None,
    )(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Softmax(name="birdSub")(x)

    # Create new model
    model = tf.keras.Model(inputs=model.input, outputs=outputs + [x])

    # Freeze all layers
    for layer in model.layers:
        layer.trainable = False

    # Thaw layers
    for layer in thaw_layers:
        model.get_layer(layer).trainable = True

    if basic_weights is None or sub_weights is None:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=lr, epsilon=0.1),
            loss=[
                tf.keras.losses.CategoricalCrossentropy(),
                tf.keras.losses.CategoricalCrossentropy(),
            ],
            metrics=["accuracy", "top_k_categorical_accuracy"],
            loss_weights=loss_weights,
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=0.1),
            loss=[
                weighted_cce(basic_weights),
                weighted_cce(sub_weights),
            ],
            metrics=["accuracy", "top_k_categorical_accuracy"],
            loss_weights=loss_weights,
        )

    model.summary()

    # Train model
    fit = model.fit(
        trainDs,
        epochs=epochs,
        validation_data=valDs,
        callbacks=callbacks,
    )

    return fit


def train_control_model(
    model,
    trainDs,
    valDs,
    lr,
    epochs,
    basic_weights=None,
    thaw_layers=["fc7", "fc8"],
    callbacks=[],
):
    """
    Train a model for a few extra epochs without any manipulations to the
    original model.
    """
    # Freeze all layers
    for layer in model.layers:
        layer.trainable = False

    # Thaw layers
    if "all" in thaw_layers:
        for layer in model.layers:
            layer.trainable = True
    else:
        for layer in thaw_layers:
            model.get_layer(layer).trainable = True

    if basic_weights is None:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=lr, epsilon=0.1, weight_decay=0.0005
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy", "top_k_categorical_accuracy"],
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=lr, epsilon=0.1, weight_decay=0.0005
            ),
            loss=weighted_cce(basic_weights),
            metrics=["accuracy", "top_k_categorical_accuracy"],
        )

    # Train model
    fit = model.fit(
        trainDs,
        epochs=epochs,
        validation_data=valDs,
        callbacks=callbacks,
    )

    return fit, model


def train_finetune_model(
    model,
    trainDs,
    valDs,
    lr,
    epochs,
    basic_weights=None,
    thaw_layers=["fc7", "fc8"],
    callbacks=[],
):
    """
    Train a model following the basic fine tuning procedures.
    """
    # Freeze all layers
    for layer in model.layers:
        layer.trainable = False

    # Thaw layers
    if "all" in thaw_layers:
        for layer in model.layers:
            layer.trainable = True
    else:
        for layer in thaw_layers:
            model.get_layer(layer).trainable = True

    if basic_weights is None:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=lr, epsilon=0.1, weight_decay=0.0005
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy", "top_k_categorical_accuracy"],
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=lr, epsilon=0.1, weight_decay=0.0005
            ),
            loss=weighted_cce(basic_weights),
            metrics=["accuracy", "top_k_categorical_accuracy"],
        )

    # Train model
    fit = model.fit(
        trainDs,
        epochs=epochs,
        validation_data=valDs,
        callbacks=callbacks,
    )

    return fit, model


class TwoHotSubAccuracy(tf.keras.metrics.Metric):
    def __init__(self, sub_nodes=200, top_k=1, name="sub_accuracy", **kwargs):
        super(TwoHotSubAccuracy, self).__init__(name=name, **kwargs)
        self.top_k = top_k
        self.sub_nodes = sub_nodes
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Reshape to ensure a batch dimensions
        y_true = tf.reshape(y_true, (-1, y_true.shape[-1]))
        y_pred = tf.reshape(y_pred, (-1, y_pred.shape[-1]))

        # Only keep the last subordinate nodes
        y_true = y_true[:, -self.sub_nodes :]
        y_pred = y_pred[:, -self.sub_nodes :]

        # Find the samples with two-hot
        trueSums = tf.reduce_sum(y_true, axis=1)
        subIndices = tf.where(tf.greater(trueSums, 0))
        subIndices = tf.squeeze(subIndices)

        if tf.size(subIndices) != 0:
            # Get the true and predicted labels for those samples
            y_true = tf.gather(y_true, subIndices)
            y_pred = tf.gather(y_pred, subIndices)

            # Get labels
            y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)
            y_pred = tf.math.top_k(y_pred, k=self.top_k, sorted=True).indices
            y_pred = tf.transpose(y_pred)

            # Calculate accuracy
            correct = tf.cast(tf.equal(y_pred, y_true), tf.float32)
            self.correct.assign_add(tf.reduce_sum(correct))

            self.count.assign_add(tf.cast(tf.size(subIndices), tf.float32))

    @tf.function
    def result(self):
        return (
            self.correct / self.count
            if self.count != 0
            else tf.constant(0, dtype=tf.float32)
        )


class OneHotBirdAccuracy(tf.keras.metrics.Metric):
    def __init__(self, top_k=1, name="bird_accuracy", **kwargs):
        super(OneHotBirdAccuracy, self).__init__(name=name, **kwargs)
        self.top_k = top_k
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Only keep the last 200 classes
        y_true = y_true[:, -200:]
        y_pred = y_pred[:, -200:]

        # Find the samples that ar birds
        trueSums = tf.reduce_sum(y_true, axis=-1)
        birdIndices = tf.where(tf.equal(trueSums, 1))
        birdIndices = tf.squeeze(birdIndices)

        if tf.size(birdIndices) != 0:
            # Get the true and predicted labels for those samples
            y_true = tf.gather(y_true, birdIndices)
            y_pred = tf.gather(y_pred, birdIndices)

            # Get labels
            y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)
            y_pred = tf.math.top_k(y_pred, k=self.top_k, sorted=True).indices
            y_pred = tf.transpose(y_pred)

            # Calculate accuracy
            correct = tf.cast(tf.equal(y_pred, y_true), tf.float32)
            self.correct.assign_add(tf.reduce_sum(correct))

            self.count.assign_add(tf.cast(tf.size(birdIndices), tf.float32))

    @tf.function
    def result(self):
        return (
            self.correct / self.count
            if self.count != 0
            else tf.constant(0, dtype=tf.float32)
        )


def validate_sub_dataset(model, directory, category):
    """
    Test the accuracy of the model on the subordinate category contained in
    the directory.
    """
    # list folders inside directory (and sort)
    classes = os.listdir(directory)
    classes.sort()

    # Find what index the category is
    categoryIdx = classes.index(category)

    # List the subclasses inside category
    subClasses = os.listdir(os.path.join(directory, category))
    subClasses.sort()

    # Create a flat dataset from the category
    ds, _ = create_flat_dataset(
        os.path.join(directory, category),
        size=224,
        channel_first=False,
        batch_size=32,
    )

    # Loop through batches
    nCorrect = 0
    nIncorrect = 0
    incorrectCats = np.empty(0)
    for x, y in ds:
        y = tf.argmax(y, axis=-1)

        # Predict x
        y_pred = model.predict(x)

        # Check how many match the category index
        correct = tf.equal(tf.argmax(y_pred, axis=-1), categoryIdx)
        nCorrect += np.sum(correct)
        nIncorrect += np.sum(~correct)

        # If there are any incorrect
        if np.sum(~correct) > 0:
            # Get the incorrect categories
            incorrectCats = np.append(incorrectCats, y[~correct])

    # Print results
    print(f"Correct: {nCorrect}")
    print(f"Incorrect: {nIncorrect}")
    print(f"Accuracy: {nCorrect / (nCorrect + nIncorrect)}")

    # Get counts of incorrectCats
    uniqueVals, counts = np.unique(incorrectCats, return_counts=True)
    uniqueVals = uniqueVals.astype(int)

    # Sort by counts
    sortIdx = np.argsort(counts)[::-1]
    uniqueVals = uniqueVals[sortIdx]
    counts = counts[sortIdx]

    # Print
    print("Incorrect counts:")
    for val, count in zip(uniqueVals, counts):
        print(f"{subClasses[val]}: {count}")

    return uniqueVals, counts


def create_level_dataset(
    directory, level, size=224, channel_first=False, batch_size=32
):
    """
    Return a dataset given a nested directory structure to specifically train at
    a specific level.
    """
    # Go through the nested directory and fill in categoryLevels
    categoryLevels = {}

    # Fill in level 1 first
    tmp = [
        file for file in glob.glob(os.path.join(directory, "*")) if os.path.isdir(file)
    ]
    tmp.sort()
    categoryLevels[1] = tmp

    curLevel = 1
    while True:
        categories = categoryLevels[curLevel]

        levelCats = []
        for cat in categories:
            levelCats += [
                file
                for file in glob.glob(os.path.join(cat, "*"))
                if os.path.isdir(file)
            ]
        levelCats.sort()

        # Check if levelCats is empty
        if len(levelCats) == 0:
            break

        # Incrememnt curLevel and save
        curLevel += 1
        categoryLevels[curLevel] = levelCats

    # Get the categories at level
    categories = categoryLevels[level]

    categoryCounts = np.array([])
    paths = []
    labels = []
    for i, cat in enumerate(categories):
        files = glob.glob(os.path.join(cat, "**/*.jpg"), recursive=True)
        nFiles = len(files)

        paths += files
        labels += [i] * nFiles

        categoryCounts = np.append(categoryCounts, nFiles)

    paths = tf.convert_to_tensor(paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    # Calculate class weights
    weights = np.sum(categoryCounts) / (len(categories) * categoryCounts)
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

        # One-hot encode labels
        y = tf.one_hot(y, len(categories))

        return x, y

    ds = (
        tf.data.Dataset.from_tensor_slices((paths, labels))
        .shuffle(len(paths))
        .map(_parse_image)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Print dataset info
    print(f"Found {len(paths)} images belonging to {len(categories)} classes.")

    return ds, weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a model for the deep cats project."
    )
    parser.add_argument(
        "--script",
        type=str,
        help="type of model to train",
        choices=["ecoCubAmnesia", "twoHot", "branch", "control", "finetune", "finetune"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed to use for training",
    )
    parser.add_argument(
        "--learningRate",
        type=float,
        help="initial learning rate to use for training",
        default=0.001,
    )
    parser.add_argument(
        "--lrDecay",
        type=float,
        help="learning rate decay factor",
        default=1.0,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="number of epochs to train for",
        default=10,
    )
    parser.add_argument(
        "--augment",
        type=bool,
        help="whether to use data augmentation",
        default=False,
    )
    parser.add_argument(
        "--batchNorm",
        help="whether to use batch normalization",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--dataDir",
        type=str,
        help="directory containing the training data",
        default="./images/ecoset_nestedSub",
    )
    parser.add_argument(
        "--sub_class",
        type=str,
        help="directory that contains the subordinate classes",
        default="0085_bird",
    )
    parser.add_argument(
        "--thaw_layers",
        type=str,
        nargs="+",
        help="layers to thaw",
        default=["fc7", "fc8"],
    )
    parser.add_argument(
        "--l2_reg",
        help="whether to use l2 regularization",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--new_weights",
        help="use new weights",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--softmax_labels",
        help="use softmax labels for multiple labels",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--activation",
        type=str,
        help="activation function to use",
        default="softmax",
        choices=["softmax", "sigmoid", "zero_hot"],
    )
    parser.add_argument(
        "--sub_loss_weight",
        type=float,
        help="weight of the sub loss",
        default=0.5,
    )
    parser.add_argument(
        "--sub_layers",
        type=int,
        help="number of extra layers to add to the sub branch",
        default=0,
    )
    parser.add_argument(
        "--pretrain",
        help="whether to pretrain the model",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        help="which gpu to use",
        default=None,
    )
    parser.add_argument(
        "--debug",
        help="whether to use debug mode",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    # If a gpu id is given, use that gpu
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        strategy = tf.distribute.get_strategy()
    else:
        strategy = tf.distribute.MirroredStrategy()

    seed = args.seed if args.seed is not None else 2023
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    tf.config.run_functions_eagerly(args.debug)

    if args.script == "ecoCubAmnesia":
        augment = args.augment
        batchNorm = args.batchNorm
        dataDir = args.dataDir

        # Training seed
        size = 256 if augment else 224

        # Create dataset
        trainDs, weights = create_flat_dataset(
            os.path.join(dataDir, "train"), size=size
        )
        valDs, _ = create_flat_dataset(
            os.path.join(dataDir, "val"),
            size=size,
        )

        weightPath = f"./models/AlexNet/ecoset_training_seeds_01_to_10/training_seed_{seed:02}/model.ckpt_epoch89"
        model = ecoset.make_alex_net_v2(
            weights_path=weightPath,
            softmax=True,
            augment=augment,
            input_shape=(size, size, 3),
            l2_reg=args.l2_reg,
        )

        # Make callbacks
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f"./models/deepCats/AlexNet/ecoCUBAmnesia/seed{seed:02}/epoch{{epoch:02d}}-val_loss{{val_loss:.2f}}.hdf5",
            monitor="val_loss",
            save_freq="epoch",
        )

        hyperParams = (
            f"-lr{args.learningRate}"
            f"-decay{args.lrDecay}"
            f"{'-new_weights' if args.new_weights else ''}"
            f"{'-l2_reg' if args.l2_reg else ''}"
        )
        loggingFile = f"./models/deepCats/AlexNet/ecoCUBAmnesia/seed{seed:02}/training{hyperParams}.csv"
        print("Logging to ", loggingFile)
        csvLogger = tf.keras.callbacks.CSVLogger(
            loggingFile,
            append=True,
        )
        callbacks = [checkpoint, csvLogger]

        if args.lrDecay < 1.0:
            schedule = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=args.lrDecay, patience=1, verbose=1
            )
            callbacks.append(schedule)

        # Train model
        fit = train_ecocub_model(
            model=model,
            trainDs=trainDs,
            valDs=valDs,
            class_weights=weights,
            lr=args.learningRate,
            epochs=args.epochs,
            callbacks=callbacks,
            batch_norm=batchNorm,
            l2_reg=args.l2_reg,
            reuse_weights=not args.new_weights,
        )
    elif args.script == "twoHot":
        weightPath = f"./models/AlexNet/ecoset_training_seeds_01_to_10/training_seed_{seed:02}/model.ckpt_epoch89"
        with strategy.scope():
            if args.pretrain:
                modelReady = False
                # Look for a control model with the first epoch done
                controlPath = os.path.join(
                    f"./models/deepCats/AlexNet/control/seed{seed:02}"
                )
                if os.path.exists(controlPath):
                    # List files
                    files = os.listdir(controlPath)
                    epochs = [file for file in files if "epoch" in file]
                    epochs.sort()
                    if len(epochs) > 0:
                        # Load the last model
                        model = tf.keras.models.load_model(
                            os.path.join(controlPath, epochs[-1]), compile=False
                        )
                        print("Loaded model from ", epochs[-1])
                        modelReady = True

                if not modelReady:
                    print("Training model for 1 epoch without subordinate category")
                    model = ecoset.make_alex_net_v2(
                        weights_path=weightPath,
                        input_shape=(224, 224, 3),
                        l2_reg=args.l2_reg,
                        softmax=True,
                    )

                    # Create dataset
                    trainDs, weights = create_flat_dataset(
                        os.path.join(args.dataDir, "train"), size=224
                    )
                    valDs, _ = create_flat_dataset(
                        os.path.join(args.dataDir, "test"),
                        size=224,
                    )

                    # Train model
                    _, model = train_control_model(
                        model=model,
                        trainDs=trainDs,
                        valDs=valDs,
                        lr=args.learningRate,
                        epochs=1,
                        thaw_layers=args.thaw_layers,
                        basic_weights=weights,
                    )

                # Remove the last layer of model
                model = tf.keras.Model(
                    inputs=model.input, outputs=model.layers[-2].output
                )
            else:
                model = ecoset.make_alex_net_v2(
                    weights_path=weightPath,
                    input_shape=(224, 224, 3),
                    l2_reg=args.l2_reg,
                )

            trainDs, weights, twoHots = create_twohot_dataset(
                os.path.join(args.dataDir, "train"),
                args.sub_class,
                size=224,
                channel_first=False,
                batch_size=32,
                softmax_labels=args.softmax_labels,
            )
            valDs, _, _ = create_twohot_dataset(
                os.path.join(args.dataDir, "test"),
                args.sub_class,
                size=224,
                channel_first=False,
                batch_size=32,
                softmax_labels=args.softmax_labels,
            )

            # Make callbacks
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                f"./models/deepCats/AlexNet/twoHot/seed{seed:02}/epoch{{epoch:02d}}-val_loss{{val_loss:.2f}}.hdf5",
                monitor="val_loss",
                save_freq="epoch",
            )

            hyperParams = (
                f"{args.activation}"
                f"-{args.sub_class}"
                f"-lr{args.learningRate}"
                f"-decay{args.lrDecay}"
                f"{'-new_weights' if args.new_weights else ''}"
                f"{'-softmax_labels' if args.softmax_labels else ''}"
                f"{'-l2_reg' if args.l2_reg else ''}"
                f"{'-pretrained' if args.pretrain else ''}"
                f"{'-sub_layers'+str(args.sub_layers) if args.sub_layers > 0 else ''}"
            )
            thawed = "-thaw"
            for layer in args.thaw_layers:
                thawed += f"{layer}"

            hyperParams += thawed
            loggingFile = f"./models/deepCats/AlexNet/twoHot/seed{seed:02}/training-{hyperParams}.csv"
            print("Logging to ", loggingFile)
            csvLogger = tf.keras.callbacks.CSVLogger(loggingFile, append=True)
            callbacks = [checkpoint, csvLogger]

            if args.lrDecay < 1.0:
                schedule = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=args.lrDecay, patience=1, verbose=1
                )
                callbacks.append(schedule)

            # Train model
            fit = train_twohot_model(
                model=model,
                trainDs=trainDs,
                valDs=valDs,
                lr=args.learningRate,
                epochs=args.epochs,
                two_hots=twoHots,
                class_weights=weights,
                batch_norm=args.batchNorm,
                sub_layers=args.sub_layers,
                callbacks=callbacks,
                thaw_layers=args.thaw_layers,
                l2_reg=args.l2_reg,
                softmax=args.activation == "softmax",
                reuse_weights=not args.new_weights,
            )
    elif args.script == "branch":
        # Weight path
        weightPath = f"./models/AlexNet/ecoset_training_seeds_01_to_10/training_seed_{seed:02}/model.ckpt_epoch89"

        with strategy.scope():
            model = ecoset.make_alex_net_v2(
                weights_path=weightPath,
                input_shape=(224, 224, 3),
                softmax=True,
                l2_reg=args.l2_reg,
            )

            # Create datasets
            trainDs, basicWeights, subWeights = create_nested_dataset(
                os.path.join(args.dataDir, "train"),
                size=224,
                not_birds=args.activation != "zero_hot",
            )
            valDs, _, _ = create_nested_dataset(
                os.path.join(args.dataDir, "val"),
                size=224,
                not_birds=args.activation != "zero_hot",
            )

            # Make callbacks
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                f"./models/deepCats/AlexNet/branch/seed{seed:02}/epoch{{epoch:02d}}-val_loss{{val_loss:.2f}}.hdf5",
                monitor="val_loss",
                save_freq="epoch",
            )

            hyperParams = (
                f"-lr{args.learningRate}"
                f"-decay{args.lrDecay}"
                f"-sub_loss_weight{args.sub_loss_weight}"
                f"-{args.activation}"
                f"{'-l2_reg' if args.l2_reg else ''}"
            )

            thawed = "-thaw"
            for layer in args.thaw_layers:
                thawed += f"{layer}"
            hyperParams += thawed

            loggingFile = f"./models/deepCats/AlexNet/branch/seed{seed:02}/training-{hyperParams}.csv"
            print("Logging to ", loggingFile)
            csvLogger = tf.keras.callbacks.CSVLogger(loggingFile, append=True)
            callbacks = [checkpoint, csvLogger]

            if args.lrDecay < 1.0:
                schedule = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=args.lrDecay, patience=1, verbose=1
                )
                callbacks.append(schedule)

            loss_weight = [1 - args.sub_loss_weight, args.sub_loss_weight]
            fit = train_branch_model(
                model=model,
                trainDs=trainDs,
                valDs=valDs,
                thaw_layers=args.thaw_layers,
                basic_weights=basicWeights,
                sub_weights=subWeights,
                lr=args.learningRate,
                epochs=args.epochs,
                l2_reg=args.l2_reg,
                non_bird_node=args.activation != "zero_hot",
                callbacks=callbacks,
                loss_weights=loss_weight,
            )
    elif args.script == "control":
        dataDir = args.dataDir

        # Create dataset
        trainDs, weights = create_flat_dataset(os.path.join(dataDir, "train"), size=224)
        valDs, _ = create_flat_dataset(
            os.path.join(dataDir, "test"),
            size=224,
        )

        weightPath = f"./models/AlexNet/ecoset_training_seeds_01_to_10/training_seed_{seed:02}/model.ckpt_epoch89"
        model = ecoset.make_alex_net_v2(
            weights_path=weightPath,
            softmax=True,
            input_shape=(224, 224, 3),
            l2_reg=args.l2_reg,
        )

        # Make callbacks
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f"./models/deepCats/AlexNet/control/seed{seed:02}/epoch{{epoch:02d}}-val_loss{{val_loss:.2f}}.hdf5",
            monitor="val_loss",
            save_freq="epoch",
        )

        hyperParams = (
            f"-lr{args.learningRate}"
            f"-decay{args.lrDecay}"
            f"{'-new_weights' if args.new_weights else ''}"
            f"{'-l2_reg' if args.l2_reg else ''}"
        )
        loggingFile = (
            f"./models/deepCats/AlexNet/control/seed{seed:02}/training{hyperParams}.csv"
        )
        print("Logging to ", loggingFile)
        csvLogger = tf.keras.callbacks.CSVLogger(
            loggingFile,
            append=True,
        )
        callbacks = [checkpoint, csvLogger]

        if args.lrDecay < 1.0:
            schedule = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=args.lrDecay, patience=1, verbose=1
            )
            callbacks.append(schedule)

        # Train model
        fit, _ = train_control_model(
            model=model,
            trainDs=trainDs,
            valDs=valDs,
            lr=args.learningRate,
            epochs=args.epochs,
            callbacks=callbacks,
            thaw_layers=args.thaw_layers,
            basic_weights=weights,
        )

    elif args.script == "finetune":
        dataDir = args.dataDir

        # Create dataset
        trainDs, weights = create_flat_dataset(os.path.join(dataDir, "train"), size=224)
        valDs, _ = create_flat_dataset(
            os.path.join(dataDir, "test"),
            size=224,
        )

        weightPath = f"./models/AlexNet/ecoset_training_seeds_01_to_10/training_seed_{seed:02}/model.ckpt_epoch89"
        model = ecoset.make_alex_net_v2(
            weights_path=weightPath,
            softmax=True,
            input_shape=(224, 224, 3),
            l2_reg=args.l2_reg,
        )

        # Modify model for new classes
        out = model.get_layer("fc7").output
        out = tf.keras.layers.Conv2D(
            weights.shape[0],
            (1, 1),
            padding="same",
            activation=None,
            kernel_regularizer=(
                tf.keras.regularizers.l2(0.0005) if args.l2_reg else None
            ),
            name="fc8",
        )(out)
        out = tf.keras.layers.GlobalAveragePooling2D()(out)
        out = tf.keras.layers.Flatten()(out)
        out = tf.keras.layers.Softmax()(out)
        model = tf.keras.Model(inputs=model.input, outputs=out)

        # Make callbacks
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f"./models/deepCats/AlexNet/finetune/seed{seed:02}/epoch{{epoch:02d}}-val_loss{{val_loss:.2f}}.hdf5",
            monitor="val_loss",
            save_freq="epoch",
        )

        hyperParams = (
            f"-lr{args.learningRate}"
            f"-decay{args.lrDecay}"
            f"{'-new_weights' if args.new_weights else ''}"
            f"{'-l2_reg' if args.l2_reg else ''}"
        )
        loggingFile = f"./models/deepCats/AlexNet/finetune/seed{seed:02}/training{hyperParams}.csv"
        print("Logging to ", loggingFile)
        csvLogger = tf.keras.callbacks.CSVLogger(
            loggingFile,
            append=True,
        )
        callbacks = [checkpoint, csvLogger]

        if args.lrDecay < 1.0:
            schedule = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=args.lrDecay, patience=1, verbose=1
            )
            callbacks.append(schedule)

        # Train model
        fit, _ = train_control_model(
            model=model,
            trainDs=trainDs,
            valDs=valDs,
            lr=args.learningRate,
            epochs=args.epochs,
            callbacks=callbacks,
            thaw_layers=args.thaw_layers,
            basic_weights=weights,
        )
    else:  # Main script
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        strategy = tf.distribute.get_strategy()

        with strategy.scope():
            level = 3
            trainDs, weights = create_level_dataset("./images/deepLevels/train", 2)
            valDs, _ = create_level_dataset("./images/deepLevels/test", 2)

            # Load model
            weightPath = f"./models/AlexNet/ecoset_training_seeds_01_to_10/training_seed_01/model.ckpt_epoch89"
            model = ecoset.make_alex_net_v2(
                weights_path=weightPath, input_shape=(224, 224, 3), softmax=True
            )

            # Validate dataset
            validate_sub_dataset(
                model, "./images/ecoset_nestedSub/test", "0356_airplane"
            )
