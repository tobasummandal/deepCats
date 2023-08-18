import os
import PIL.Image as Image
import numpy as np
import tensorflow as tf
import pandas as pd
from itertools import combinations


# List folders
def build_categories_from_ecoset(path, synsets=None, maxImgs=None, includeSub=True):
    """
    Return dictionary of the images and counts for each category in path.
    Only keep synsets in the list synsets if defined and only keep the first
    maxImgs images.
    """
    cats = {}
    for name in os.listdir(path):
        if os.path.isdir(os.path.join(path, name)):
            cats[name] = {}
            # List folders in that folder
            for name2 in os.listdir(os.path.join(path, name)):
                if os.path.isdir(os.path.join(path, name, name2)):
                    if includeSub:
                        cats[name][name2] = {}
                        # List images in that folder
                        images = os.listdir(os.path.join(path, name, name2))
                        # Split images into their categories
                        subCats = [img.split("_")[0] for img in images]

                        # Keep unique
                        subCats = list(set(subCats))

                        # Filter synsets
                        if synsets is not None:
                            subCats = [
                                subCat for subCat in subCats if subCat in synsets
                            ]

                        # Fill dictionary with each category and its images
                        for subCat in subCats:
                            imgs = [
                                os.path.join(path, name, name2, img)
                                for img in images
                                if subCat in img
                            ]

                            # Keep only the first maxImgs images
                            if maxImgs is not None:
                                imgs = imgs[:maxImgs]

                            cats[name][name2][subCat] = imgs
                    else:
                        imgs = os.listdir(os.path.join(path, name, name2))

                        if synsets is not None:
                            imgs = [img for img in imgs if img.split("_")[0] in synsets]

                        if maxImgs is not None:
                            imgs = imgs[:maxImgs]

                        cats[name][name2] = [
                            os.path.join(path, name, name2, img) for img in imgs
                        ]

    # Get counts of each image in each category
    counts = {}
    for cat in cats:
        counts[cat] = {}
        for subCat in cats[cat]:
            if includeSub:
                counts[cat][subCat] = {}
                for img in cats[cat][subCat]:
                    counts[cat][subCat][img] = len(cats[cat][subCat][img])
            else:
                counts[cat][subCat] = len(cats[cat][subCat])

    return cats, counts


def build_df_from_dir(directory, cats=[]):
    """
    Return a dataframe where each row is an image recursively where each row
    is an image with extra columns based on how deep in the directory structure
    it is.
    """
    # Create pandas dataframe
    df = pd.DataFrame(
        columns=["path", "name", "cat1"] + [f"cat{i + 2}" for i in range(len(cats))]
    )

    fileList = os.listdir(directory)
    # Ignore hidden files
    fileList = [file for file in fileList if not file.startswith(".")]

    for name in fileList:
        if os.path.isdir(os.path.join(directory, name)):
            newRow = build_df_from_dir(
                os.path.join(directory, name), cats=cats + [name]
            )
            newRow
        else:
            newRow = pd.DataFrame(
                [[os.path.join(directory, name), name] + cats + [name]],
                columns=["path", "name", "cat1"]
                + [f"cat{i + 2}" for i in range(len(cats))],
            )

        df = pd.concat([df, newRow], sort=False, ignore_index=True)

    # If name is equal to the last column, remove last column
    if df["name"].equals(df.iloc[:, -1]):
        df = df.iloc[:, :-1]

    return df


def get_images_from_cat(cats, preprocFun=None):
    """
    Return a dictionary of loaded images from path recursively.
    """
    imgs = {}
    for key, values in cats.items():
        if isinstance(values, dict):
            imgs[key] = get_images_from_cat(values, preprocFun=preprocFun)
        else:
            tmp = [Image.open(img) for img in values]
            if preprocFun is not None:
                tmp = [preprocFun(img) for img in tmp]
                # Stack tmp
                tmp = tf.concat(tmp, axis=0)

            imgs[key] = tmp

    return imgs


def gcm_sim(rep1, rep2, r=2.0, c=1.0, p=1.0):
    """
    Return the GCM similarity between two representations with equal attention
    weights.
    """
    assert np.all(rep1.shape == rep2.shape)

    weights = np.ones(rep1.shape[0]) / rep1.shape[0]

    dist = np.sum(weights * (np.abs(rep1 - rep2) ** r)) ** (1.0 / r)

    return np.exp(-c * dist**p)


def prod_sim(rep1, rep2):
    """
    Return the similarity between two representations using the product rule.
    """
    assert np.all(rep1.shape == rep2.shape)

    # Copy reps
    rep1 = rep1.copy()
    rep2 = rep2.copy()

    # Normalize both reps between 0 and 1
    rep1 = rep1 / np.sum(rep1)
    rep2 = rep2 / np.sum(rep2)

    # Compute absolute difference
    diff = np.abs(rep1 - rep2)

    # Flip such that 1 is perfectly matched
    diff = np.abs(diff - 1)

    return np.prod(diff)


def prod_sim_binary(rep1, rep2, s=0.3, threshold=0.5):
    """
    Return the simliarity between two representations using the product rule
    after binarizing the representations based on similarity threshold.
    """
    assert np.all(rep1.shape == rep2.shape)

    # Copy reps
    rep1 = rep1.copy()
    rep2 = rep2.copy()

    # Normalize both reps between 0 and 1
    rep1 = rep1 / np.sum(rep1)
    rep2 = rep2 / np.sum(rep2)

    # Compute absolute difference
    diff = np.abs(rep1 - rep2)

    # Flip such that 1 is perfectly matched
    diff = np.abs(diff - 1)

    # Binarize
    diff[diff < threshold] = s

    return np.prod(diff)


def contrast_sim(rep1, rep2, threshold=0.0):
    """
    Return similarity based on the contrast model where a feature is not
    present if the value of a feature is equal or less than the threshold.
    Uses equal weighting for overlap and distinct features for each
    representation.
    """
    assert np.all(rep1.shape == rep2.shape)

    # Copy reps
    rep1 = rep1.copy()
    rep2 = rep2.copy()

    # Binarize reps
    rep1[rep1 <= threshold] = 0
    rep2[rep2 <= threshold] = 0
    rep1[rep1 > threshold] = 1
    rep2[rep2 > threshold] = 1

    # Compute overlap
    overlap = np.sum(rep1 * rep2)

    # Count distinct feature
    distinct = np.sum(np.abs(rep1 - rep2))

    if (sim := overlap - distinct) < 0:
        sim = 0.0

    return sim


def calculate_typicality(reps, simFun, nExemplars=None):
    """
    Return the typicality of each item given a category defined by the
    representation.
    """
    if isinstance(reps, list):
        reps = np.concatenate(reps, axis=0)

    typicalities = np.empty(reps.shape[0])
    for i, rep in enumerate(reps):
        # Remove rep row from reps
        reps_ = reps.copy()
        reps_ = np.delete(reps_, i, axis=0)

        if nExemplars is not None:
            # Keep only nExemplars
            reps_ = reps_[np.random.choice(reps_.shape[0], nExemplars, replace=False)]

        # Calculate typicality
        typ = np.sum(np.apply_along_axis(lambda x: simFun(rep, x), 1, reps_))

        typicalities[i] = typ

    return typicalities


def feature_select(rep, b=0.0, d=0.8):
    """
    Return an logical array for the features selected in rep based on the
    threshold activation b and the a threshold porotion of d.
    """
    # Binarize representation
    rep = rep > b

    # Sum features across samples
    repCount = np.sum(rep, axis=0)

    # Return which features exceed proportion
    return repCount > (rep.shape[0] * d)


def redist_evidence(
    testRep, targetRep, altRep, simFun, b=0.0, d=0.8, dist_penalty=False
):
    """
    Calculate evidence that testRep is the category targetRep against the
    alternative altRep given redundancy and distinctiveness using a threshold
    activation b and a threshold proportion of d.
    """
    # Determine what features each category has
    targetFeatures = feature_select(targetRep, b=b, d=d)

    # Filter for selected features in the test representation
    testRepSelected = testRep[targetFeatures]
    targetRepsSelected = targetRep[:, targetFeatures]

    # Calculate similarity between test and target representations
    sim = np.sum(
        np.apply_along_axis(lambda x: simFun(x, testRepSelected), 1, targetRepsSelected)
    )

    if dist_penalty:
        altFeatures = feature_select(altRep, b=b, d=d)

        # Find conjunctions between target and alternative
        overlap = np.logical_and(targetFeatures, altFeatures)

        # Select features that are overlapped
        testOverlapRep = testRep[overlap]
        altOverlapRep = altRep[:, overlap]

        # Calculate similarity between test and alternative
        distPenalty = np.sum(
            np.apply_along_axis(lambda x: simFun(x, testOverlapRep), 1, altOverlapRep)
        )

        sim = sim - distPenalty

    return sim


def sim_prob(rep, cat1Rep, cat2Rep, simFun, equalize=False, nExemplars=None):
    """
    Return a probability of responding one of two categories represented by
    cat1Rep and cat2Rep to a given representation rep. Assumes that the first
    dimension of category representations are each exemplar and the second
    dimension are features. If either category representations are a list,
    concatenate them. If equalize, then the number of exemplar for each
    category is equalized by randomly sampling without replacement from the
    larger category equal to the smaller category. If nExemplars is not None,
    limit the exemplar counts in each representation to nExemplars by random
    sampling.
    """
    if isinstance(cat1Rep, list):
        cat1Rep = np.concatenate(cat1Rep, axis=0)
    if isinstance(cat2Rep, list):
        cat2Rep = np.concatenate(cat2Rep, axis=0)

    if nExemplars is not None and nExemplars < cat1Rep.shape[0]:
        cat1Rep = cat1Rep[np.random.choice(cat1Rep.shape[0], nExemplars, False)]

    if nExemplars is not None and nExemplars < cat2Rep.shape[0]:
        cat2Rep = cat2Rep[np.random.choice(cat2Rep.shape[0], nExemplars, False)]

    if equalize:
        if cat1Rep.shape[0] < cat2Rep.shape[0]:
            cat2Rep = cat2Rep[
                np.random.choice(cat2Rep.shape[0], cat1Rep.shape[0], replace=False)
            ]
        else:
            cat1Rep = cat1Rep[
                np.random.choice(cat1Rep.shape[0], cat2Rep.shape[0], replace=False)
            ]

    rep1 = np.sum(np.apply_along_axis(lambda x: simFun(rep, x), 1, cat1Rep))
    rep2 = np.sum(np.apply_along_axis(lambda x: simFun(rep, x), 1, cat2Rep))

    return rep1 / (rep1 + rep2), rep2 / (rep1 + rep2)


def LBA_deterministic(d1, d2, k=0, b=1, t0=0):
    """
    Return response and response time for a 2 alternate decision task where
    each accumulator only differ in their drift rate
    """
    rt1 = ((b - k) / d1) + t0
    rt2 = ((b - k) / d2) + t0

    if rt1 < rt2:
        return 1, rt1
    else:
        return 2, rt2


def get_evidence(rep, catRep, simFun, maxExemplars=None):
    """
    Return evidence for this catRep given the rep using simFun. If maxExemplars
    is not None, then limit the number of exemplars in the category.
    """
    if maxExemplars is not None and maxExemplars < catRep.shape[0]:
        choices = np.random.choice(catRep.shape[0], maxExemplars, replace=False)
        catRep = catRep[choices]

    return np.sum(np.apply_along_axis(lambda x: simFun(rep, x), 1, catRep))


def simulate_cat_verification(
    testReps,
    memoryReps,
    testImgInfo,
    memoryImgInfo,
    categoryCol,
    modelName,
    simFun,
    criterion,
    maxImgs=None,
    catRepIdxs=None,
):
    """
    Return a dataframe simulating the results of a category verification task.
    """
    # Setup dataframe
    performance = pd.DataFrame(
        columns=[
            "seed",
            "model",
            "image",
            "category",
            "level",
            "response",
            "RT",
            "crit",
            "maxImgs",
        ]
    )

    # Get number of models
    nModels = len(testReps)

    # Get categories
    categories = np.unique(testImgInfo[categoryCol].dropna())

    # Loop through categories
    for category in categories:
        # Get the representations of this category
        catIdx = testImgInfo[testImgInfo[categoryCol] == category].index

        if catRepIdxs is not None:
            catIdxs = catRepIdxs[category]

        # Loop through models
        for i in range(nModels):
            # Get reps for this model
            memoryModelReps = memoryReps[i, catIdx, :]
            testModelReps = testReps[i, catIdx, :]

            if catRepIdxs is not None:
                catIdxs = np.unique(np.concatenate(catRepIdxs[category]))
                memoryModelReps = memoryModelReps[:, :, :, catIdxs]
                testModelReps = testModelReps[:, :, :, catIdxs]

            # Flatten reps
            memoryModelReps = memoryModelReps.reshape(memoryModelReps.shape[0], -1)
            testModelReps = testModelReps.reshape(testModelReps.shape[0], -1)

            # Loop through images
            for j, imgRep in enumerate(testModelReps):
                # Image rows
                imgInfo = testImgInfo.iloc[catIdx[j]]

                if maxImgs is not None:
                    catReps = memoryModelReps[
                        np.random.choice(memoryModelReps.shape[0], maxImgs, False)
                    ]
                else:
                    catReps = memoryModelReps[:]

                # Simulate trial
                evidence = get_evidence(imgRep, catReps, simFun)
                drift = evidence / (evidence + criterion)
                resp, rt = LBA_deterministic(drift, 1 - drift, b=0.5)
                resp = "yes" if resp == 1 else "no"

                # Add trial to performance df
                performance = pd.concat(
                    [
                        performance,
                        pd.DataFrame(
                            {
                                "seed": i + 1,
                                "model": modelName,
                                "image": imgInfo["name"],
                                "category": category,
                                "level": categoryCol,
                                "response": resp,
                                "RT": rt,
                                "crit": criterion,
                                "maxImgs": maxImgs,
                            },
                            index=[0],
                        ),
                    ]
                )

    return performance


def cluster_index(imgInfo, levelCol, category, imgSet, simMat, normalize=False):
    loc = (imgInfo[levelCol] == category) & (imgInfo["set"] == imgSet)
    withinIdxs = imgInfo.loc[loc, "name"].index

    loc = (imgInfo[levelCol] != category) & (imgInfo["set"] == imgSet)
    betweenIdxs = imgInfo.loc[loc, "name"].index

    if normalize:
        simMat = simMat / np.max(simMat)

    withinSum = 0
    withinCount = 0
    for i, j in combinations(withinIdxs, 2):
        withinSum += simMat[i, j]
        withinCount += 1

    betweenSum = 0
    betweenCount = 0
    for i in withinIdxs:
        for j in betweenIdxs:
            betweenSum += simMat[i, j]
            betweenCount += 1

    return (withinSum / withinCount) - (betweenSum / betweenCount)


if __name__ == "__main__":
    df = build_df_from_dir("./images/deepCats/test")
    # Save df
    df.to_csv("./deepCatsTestImages.csv", index=False)
