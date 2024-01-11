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
import pandas as pd
from nltk.corpus import wordnet as wn


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


def compute_sims_over_training(
    modelDir: str, layer: str, images: np.ndarray, custom_objects: dict = None
):
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
        print(f"Processing file: {modelFile}")

        # Load model
        model = tf.keras.models.load_model(modelFile, custom_objects=custom_objects)

        # Get output at target layer
        x = model.get_layer(layer).output

        # Compile model
        model = tf.keras.models.Model(inputs=model.input, outputs=x)

        # Get representations
        reps = model.predict(images)

        # Flatten reps
        reps = reps.reshape(reps.shape[0], -1)

        # Calculate similarity matrix using GCM (note that we're using the parameters r=2, c=1, p=1)
        simMat[i] = np.exp(
            -1
            * squareform(pdist(reps, metric="euclidean"))
            * ((1 / reps.shape[1]) ** (1 / 2))
        )

        # Cleanup so we don't run out of GPU memory
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    return simMat


def get_image_info(directory):
    """
    Recursively go through the directory and build a dataframe with the
    information about the images.
    """
    # Create dataframe
    df = pd.DataFrame(columns=["path", "name", "super", "basic", "sub", "set"])

    # Loop through train directory first
    for root, dirs, files in os.walk(os.path.join(directory, "train")):
        # Loop through files
        for file in files:
            # Ignore dstore
            if file == ".DS_Store":
                continue
            # Add to dataframe
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "path": [os.path.join(root, file)],
                            "name": [file],
                            "super": [root.split("/")[-3]],
                            "basic": [root.split("/")[-2]],
                            "sub": [root.split("/")[-1]],
                            "set": ["train"],
                        }
                    ),
                ],
                ignore_index=True,
            )

    # Loop through test directory
    for root, dirs, files in os.walk(os.path.join(directory, "test")):
        # Loop through files
        for file in files:
            # Ignore dstore
            if file == ".DS_Store":
                continue

            # Add to dataframe
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "path": [os.path.join(root, file)],
                            "name": [file],
                            "super": [root.split("/")[-3]],
                            "basic": [root.split("/")[-2]],
                            "sub": [root.split("/")[-1]],
                            "set": ["test"],
                        }
                    ),
                ],
                ignore_index=True,
            )

    return df


def get_category_nodes(data_dir, img_info):
    """
    Return a dictionary of indices for all levels of the hierarchy derived from
    img_info by looking in data_dir.
    """
    category_nodes = {}

    # List folders in data_dir
    cats = os.listdir(data_dir)
    cats.sort()

    # Start at the basic level for imgInfo
    basicCats = img_info["basic"].unique()

    # For each basic cats, find the node
    for cat in basicCats:
        nodes = [i for i, folder in enumerate(cats) if cat in folder]

        if len(nodes) == 0:
            raise ValueError(f"Could not find node for {cat}")
        elif len(nodes) > 1:
            print(f"Found multiple nodes for {cat}")
            print(f"Nodes: {nodes}")
            print(f"Labels: {[cats[i] for i in nodes]}")
            choice = input("Which one to use (1, 2, ... n) ? ")
            category_nodes[cat] = nodes[int(choice) - 1]
        else:
            category_nodes[cat] = nodes[0]

    # Get all the names of the categories
    synsets = []
    for cat in cats:
        tmp = wn.synsets(cat.split("_")[-1], pos=wn.NOUN)
        if len(tmp) > 0:
            synsetList = sum(tmp[0].hypernym_paths(), [])
            synsets.append(synsetList)
        else:
            synsets.append(None)
            print("Could not find synset for ", cat)

    # For each super cat, use wordnet hierarchy to figure out valid nodes
    superCats = img_info["super"].unique()
    for cat in superCats:
        # Get the synset for the super category
        synset = wn.synsets(cat, pos=wn.NOUN)[0]
        superNodes = []
        for i, synsetList in enumerate(synsets):
            if synsetList is None:
                continue

            if synset in synsetList:
                superNodes.append(i)

        category_nodes[cat] = superNodes

    # For each subordinate cat, look in the basic cat directory to find node
    subCats = img_info["sub"].unique()
    for cat in subCats:
        basicCat = img_info.loc[img_info["sub"] == cat, "basic"].unique()[0]
        basicNode = category_nodes[basicCat]

        subNodes = os.listdir(os.path.join(data_dir, cats[basicNode]))
        subNodes.sort()

        category_nodes[cat] = subNodes.index(cat)

    return category_nodes


def find_nested_synsets(targetSynset, synsets):
    """
    Return the index and synsets from the list synsets that is nested under
    the targetSynset
    """
    nestedIdxs, nestedSynsets = [], []
    for i, synset in enumerate(synsets):
        lowest_common_hypernyms = synset.lowest_common_hypernyms(targetSynset)
        if targetSynset in lowest_common_hypernyms:
            nestedIdxs.append(i)
            nestedSynsets.append(synset)

    return nestedIdxs, nestedSynsets


def extract_training_data(directory):
    """
    Return a dataframe of training trajectory information for each model in
    directory.
    """
    # List folders in directory
    folders = os.listdir(directory)

    # Check the first folder to get the dataframe spec
    files = os.listdir(os.path.join(directory, folders[0]))
    csv = [file for file in files if file.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(directory, folders[0], csv))

    # Loop through the rest of the folders and concatenate
    for folder in folders[1:]:
        files = os.listdir(os.path.join(directory, folder))
        csv = [file for file in files if file.endswith(".csv")][0]
        df = pd.concat(
            [
                df,
                pd.read_csv(os.path.join(directory, folder, csv)),
            ],
            ignore_index=True,
        )

    return df, csv.split(".csv")[0].split("-")[1:]


if __name__ == "__main__":
    extract_training_data("./models/deepCats/AlexNet/twoHotFreezeBasic/")
