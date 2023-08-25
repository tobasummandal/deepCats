from numpy import average
import tensorflow as tf
from tensorflow.keras import layers
import os
import shutil
import glob
import numpy as np
import gc
import categorization as cat
from scipy.spatial.distance import pdist, squareform


def make_output_model(model, average_pooling=True, flatten=True):
    """Return a model that has outputs at every convolution and dense layer."""
    outputs = []
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            dataFormat = layer.data_format
            if average_pooling:
                layer = layers.GlobalAvgPool2D(data_format=dataFormat)(layer.output)

            if flatten:
                if average_pooling:
                    layer = layers.Flatten(data_format=dataFormat)(layer)
                else:
                    layer = layers.Flatten(data_format=dataFormat)(layer.output)

            if average_pooling or flatten:
                outputs.append(layer)
            else:
                outputs.append(layer.output)
        elif isinstance(layer, layers.Dense):
            outputs.append(layer.output)

    return tf.keras.Model(model.input, outputs)


def add_CUB200_data(source_dir, target_dir):
    """
    Add CUB200 data from the source_dir to the target_dir, empties bird
    directory in target_dir and copies images for the source_dir into the train
    val directories in target_dir according to the train_test_split file.
    """
    # Empty the bird directory in target_dir
    birdDir = "0085_bird"
    train_dir = os.path.join(target_dir, "train", birdDir)
    val_dir = os.path.join(target_dir, "val", birdDir)
    test_dir = os.path.join(target_dir, "test", birdDir)

    for dir in [train_dir, val_dir, test_dir]:
        for file in os.listdir(dir):
            # Recursively remove
            if os.path.isdir(os.path.join(dir, file)):
                shutil.rmtree(os.path.join(dir, file))
            else:
                os.remove(os.path.join(dir, file))

    # Load info files in source_dir
    with open(os.path.join(source_dir, "train_test_split.txt"), "r") as f:
        splitInfo = f.readlines()

    with open(os.path.join(source_dir, "images.txt"), "r") as f:
        images = f.readlines()

    # Loop through images
    for img, split in zip(images, splitInfo):
        img = img.split(" ")[1].strip()
        split = split.split(" ")[1].strip()

        # Check if intermediate directories exist
        trainDir = os.path.join(target_dir, "train", birdDir, img.split("/")[0])
        valDir = os.path.join(target_dir, "val", birdDir, img.split("/")[0])
        testDir = os.path.join(target_dir, "test", birdDir, img.split("/")[0])

        if not os.path.exists(trainDir):
            os.makedirs(trainDir)
        if not os.path.exists(valDir):
            os.makedirs(valDir)
        if not os.path.exists(testDir):
            os.makedirs(testDir)

        # Copy image to correct directory
        if split == "1":
            src = os.path.join(source_dir, "images", img)
            dst = os.path.join(train_dir, img)
            shutil.copy(src, dst)
        elif split == "0":
            src = os.path.join(source_dir, "images", img)
            valDst = os.path.join(val_dir, img)
            testDst = os.path.join(test_dir, img)
            shutil.copy(src, valDst)
            shutil.copy(src, testDst)

    return None


def split_CUB200_data(source_dir, target_dir):
    """
    Split CUB200 data from the source_dir into the train and val directories in
    target_dir according to the split file.
    """
    # Load info files in source_dir
    with open(os.path.join(source_dir, "train_test_split.txt"), "r") as f:
        splitInfo = f.readlines()

    with open(os.path.join(source_dir, "images.txt"), "r") as f:
        images = f.readlines()

    # Check if target_dir exists, make it if not
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Loop through images
    trainImgCount = 0
    valImgCount = 0
    for img, split in zip(images, splitInfo):
        img = img.split(" ")[1].strip()
        split = split.split(" ")[1].strip()
        trainVal = "train" if split == "1" else "val"

        if trainVal == "train":
            trainImgCount += 1
        else:
            valImgCount += 1

        # Check if directory exists
        className = img.split("/")[0].strip()
        targetDir = os.path.join(target_dir, trainVal, f"CUB_{className}")
        if not os.path.exists(targetDir):
            os.makedirs(targetDir)

        # Copy image to correct directory
        src = os.path.join(source_dir, "images", img)
        dst = os.path.join(targetDir, img.split("/")[-1])

        shutil.copy(src, dst)

    # Count how many categories there are
    categories = os.listdir(os.path.join(target_dir, "train"))
    nCategories = len(categories)

    # Print info
    print(f"Found {trainImgCount} training images.")
    print(f"Found {valImgCount} validation images.")
    print(f"Split across {nCategories} categories.")

    return None


def get_reps_over_training(modelDir: str, layer: str, images: np.ndarray):
    """
    Return representation from layer of the images over training from modelDir.
    We expect modelDir to have saved models over epochs.
    """
    # Get model files
    modelFiles = glob.glob(os.path.join(modelDir, "*.hdf5"))
    modelFiles.sort()

    # Load one model to figure out size
    model = tf.keras.models.load_model(modelFiles[0])
    x = model.get_layer(layer).output

    # Preallocate shape of representations
    reps = np.zeros([len(modelFiles), images.shape[0]] + list(x.shape[1:]))

    # Loop through models
    for i, modelFile in enumerate(modelFiles):
        # Cleanup so we don't run out of GPU memory
        del model
        tf.keras.backend.clear_session()
        gc.collect()

        # Load model
        model = tf.keras.models.load_model(modelFile)

        # Get output at target layer
        x = model.get_layer(layer).output

        # Compile model
        model = tf.keras.models.Model(inputs=model.input, outputs=x)

        # Get representations
        reps[i] = model.predict(images)

    return reps


def compute_sims_over_training(modelDir: str, layer: str, images: np.ndarray):
    """
    Return a similarity matrix over training from modelDir at layer using images.
    """
    # Get model files
    modelFiles = glob.glob(os.path.join(modelDir, "*.hdf5"))
    modelFiles.sort()

    # Preallocate similarity matrix
    simMat = np.zeros([len(modelFiles), images.shape[0], images.shape[0]])

    # Loop through models
    for i, modelFile in enumerate(modelFiles):
        # Load model
        model = tf.keras.models.load_model(modelFile)

        # Get output at target layer
        x = model.get_layer(layer).output

        # Compile model
        model = tf.keras.models.Model(inputs=model.input, outputs=x)

        # Get representations
        reps = model.predict(images)

        # Flatten reps
        reps = reps.reshape(reps.shape[0], -1)

        # Calculate similarity matrix
        simMat[i] = squareform(pdist(reps, metric=cat.gcm_sim))

        # Cleanup so we don't run out of GPU memory
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    return simMat


if __name__ == "__main__":
    # Load images
    images = np.load("./images/deepCatsTrainImages.npy")
    get_reps_over_training(
        modelDir="./models/deepCats/AlexNet/seed01", layer="fc7", images=images
    )
