import os
import PIL.Image as Image
import numpy as np
import tensorflow as tf
import pandas as pd
from itertools import combinations
from scipy.spatial.distance import pdist, squareform, cdist
from scipy import stats
from treelib import Tree
from HiPart.clustering import IPDDP
import sklearn.metrics as metrics


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
    weights = np.ones(rep1.shape[0]) / rep1.shape[0]

    dist = np.sum(weights * (np.abs(rep1 - rep2) ** r)) ** (1.0 / r)

    return np.exp(-c * dist**p)


def gcm_sim_thresholded(rep1, rep2, r=2.0, c=1.0, p=1.0, threshold=1):
    """
    Return the GCM similarity between two representations with equal attention
    but only calculate the distance between features that are above threshold.
    """
    # Figure out which features are above threshold
    rep1Thresh = rep1 > threshold
    rep2Thresh = rep2 > threshold

    rep1Threshed = rep1[rep1Thresh | rep2Thresh]
    rep2Threshed = rep2[rep1Thresh | rep2Thresh]

    return gcm_sim(rep1Threshed, rep2Threshed, r=r, c=c, p=p)


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


def cat_verification_from_mat(
    simMat: np.ndarray,
    imgInfo: pd.DataFrame,
    modelName: str,
    criterion: float = None,
    maxImgs: int = None,
) -> pd.DataFrame:
    """
    Simulate a category verification task given a similarity matrix with image
    info using an LBA with a criterion. If criterion is None, set the criterion
    to result in 95% accuracy.
    """
    # Find the unique categories at each level
    levelCats = {
        "super": list(np.unique(imgInfo["super"].dropna())),
        "basic": list(np.unique(imgInfo["basic"].dropna())),
        "sub": list(np.unique(imgInfo["sub"].dropna())),
    }

    # Make the index a column
    imgInfo = imgInfo.reset_index()

    # Setup dataframe
    performance = pd.DataFrame(
        columns=[
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

    # Loop through the levels
    for level in ["super", "basic", "sub"]:
        # Loop through categories in that level
        for category in levelCats[level]:
            # Get the indices of the images in the training set and test set
            trainIdxs = imgInfo[
                (imgInfo[level] == category) & (imgInfo["set"] == "train")
            ].index
            testIdxs = imgInfo[
                (imgInfo[level] == category) & (imgInfo["set"] == "test")
            ].index

            # Filter similarity matrix for only the images we need
            catSimMat = simMat[testIdxs, :][:, trainIdxs]

            if maxImgs is not None:
                # Create a new similarity matrix with only maxImgs images
                newSimMat = np.zeros((catSimMat.shape[0], maxImgs))
                for i, row in enumerate(catSimMat):
                    newSimMat[i, :] = np.random.choice(row, maxImgs, False)

                # Save over
                catSimMat = newSimMat

            evidences = np.sum(catSimMat, axis=1)

            if criterion is None:
                # Find a criterion where 95% of the time it is correct
                crit = np.quantile(evidences, 0.05)
            else:
                crit = criterion

            # Loop through the test images
            for i, evidence in enumerate(evidences):
                # Calculate drift
                drift = evidence / (evidence + crit)
                resp, rt = LBA_deterministic(drift, 1 - drift, b=0.5)
                resp = "yes" if resp == 1 else "no"

                # Add performance to dataframe
                performance = pd.concat(
                    [
                        performance,
                        pd.DataFrame(
                            {
                                "model": modelName,
                                "image": imgInfo.loc[testIdxs[i], "name"],
                                "category": category,
                                "level": level,
                                "response": resp,
                                "RT": rt,
                                "crit": crit,
                                "maxImgs": maxImgs,
                            },
                            index=[0],
                        ),
                    ]
                )

    return performance


class SimCluster:
    def __init__(self, simMat, imgInfo):
        self.simMat = simMat
        self.imgInfo = imgInfo

        # Figure out level map from imgInfo
        self.levelMap = {
            "super": list(imgInfo["super"].dropna().unique()),
            "basic": list(imgInfo["basic"].dropna().unique()),
            "sub": list(imgInfo["sub"].dropna().unique()),
        }

        # Figure out sets
        self.sets = list(imgInfo["set"].unique())

    def calculate_index(
        self, imgSet=None, level=None, category=None, within_level=False
    ):
        # Both level and category cannot be set together
        if level is not None and category is not None:
            raise ValueError("Both level and category cannot be set together")

        if imgSet is not None:
            # Filter imgInfo by sets
            imgInfo = self.imgInfo[self.imgInfo["set"] == imgSet]
        else:
            imgInfo = self.imgInfo

        # Handle average level indices first
        if level is not None:
            # Get categories
            categories = self.levelMap[level]

            # Preallocate array for cluster indices
            clusters = np.zeros(len(categories), dtype=np.float32)
            # Loop through categories
            for k, cat in enumerate(categories):
                loc = imgInfo[level] == cat
                withinIdxs = imgInfo.loc[loc, "name"].index

                if within_level and level != "super":
                    hier = list(self.levelMap.keys())
                    higherLevel = hier[hier.index(level) - 1]

                    # Get the higher level category
                    higherCat = imgInfo.loc[withinIdxs, higherLevel].unique()[0]

                    loc = (imgInfo[level] != cat) & (imgInfo[higherLevel] == higherCat)
                    betweenIdxs = imgInfo.loc[loc, "name"].index
                else:
                    loc = imgInfo[level] != cat
                    betweenIdxs = imgInfo.loc[loc, "name"].index

                withinSum = 0
                withinCount = 0
                for i, j in combinations(withinIdxs, 2):
                    withinSum += self.simMat[i, j]
                    withinCount += 1

                betweenSum = 0
                betweenCount = 0
                for i in withinIdxs:
                    for j in betweenIdxs:
                        betweenSum += self.simMat[i, j]
                        betweenCount += 1

                clusters[k] = (withinSum / withinCount) - (betweenSum / betweenCount)

            return np.mean(clusters)
        elif category is not None:
            # Find the level of the category
            for level, categories in self.levelMap.items():
                if category in categories:
                    break

            loc = imgInfo[level] == category
            withinIdxs = imgInfo.loc[loc, "name"].index

            if within_level and level != "super":
                hier = list(self.levelMap.keys())
                higherLevel = hier[hier.index(level) - 1]

                # Get the higher level category
                higherCat = imgInfo.loc[withinIdxs, higherLevel].unique()[0]

                loc = (imgInfo[level] != category) & (imgInfo[higherLevel] == higherCat)
                betweenIdxs = imgInfo.loc[loc, "name"].index
            else:
                loc = imgInfo[level] != category
                betweenIdxs = imgInfo.loc[loc, "name"].index

            withinSum = 0
            withinCount = 0
            for i, j in combinations(withinIdxs, 2):
                withinSum += self.simMat[i, j]
                withinCount += 1

            betweenSum = 0
            betweenCount = 0
            for i in withinIdxs:
                for j in betweenIdxs:
                    betweenSum += self.simMat[i, j]
                    betweenCount += 1

            return (withinSum / withinCount) - (betweenSum / betweenCount)
        else:
            raise ValueError("Either level or category must be set")

    def calculate_all(self, within_level=False):
        for imgSet in self.sets:
            for level in self.levelMap.keys():
                for category in self.levelMap[level]:
                    val = self.calculate_index(
                        imgSet=imgSet, category=category, within_level=within_level
                    )
                    print(f"{imgSet}-{level}-{category}: {val}")
            print("--")


def default_gcm_sim_mat(reps, c=1.0):
    """
    Calculate a similarity matrix using GCM with r=2, c=1, p=1.
    """
    return np.exp(
        -c
        * squareform(pdist(reps, metric="euclidean"))
        * ((1 / reps.shape[1]) ** (1 / 2))
    )


def default_gcm_cdist(reps1, reps2, c=1.0):
    """
    Calculate pairwise similarity between reps1 and reps 2 using GCM with r=2,
    c=1, p=1
    """
    return np.exp(
        -c * cdist(reps1, reps2, metric="euclidean") * ((1 / reps1.shape[1]) ** (1 / 2))
    )


def exemplar_maker(n, center, radius=1, radius_density="uniform", relu=False):
    nDims = len(center)

    # Generate random numbers as needed
    coords = np.random.normal(loc=0, scale=1, size=(n, nDims))
    uniforms = np.random.uniform(low=0, high=1, size=n)

    if radius_density == "power":
        radii = (uniforms ** (1 / nDims)) * radius
    elif radius_density == "normal":
        radii = np.abs(np.random.normal(loc=0, scale=1, size=n)) * radius
    elif radius_density == "uniform":
        radii = uniforms * radius
    elif radius_density == "lognormal":
        radii = np.random.lognormal(mean=0, sigma=1 / 3, size=n) * radius
    else:
        raise ValueError("Density type not recognized")

    coords = coords.T / np.linalg.norm(
        coords, axis=1
    )  # Uniformly distributed directions
    coords = coords * radii  # Change radii
    coords = coords.T + center

    # If relu, apply relu
    if relu:
        coords[coords < 0] = 0

    return coords


def make_categories(
    *,
    cat_rad,
    radius_density="power",
    relu=False,
    super_rad,
    basic_rad,
    sub_rad,
    nFeatures,
    nImages,
):
    def _centroids_maker(center, r):
        """
        Create two centroids on a surface of a hypersphere with radius r. The
        first centroid is randomly selected from the surface of a n-sphere
        """
        nFeatures = center.shape[0]

        coords = stats.multivariate_normal.rvs(mean=np.zeros((nFeatures,)), cov=1)

        # Change coordinates to unit length
        coords = coords / np.linalg.norm(coords)

        # Multiply coords by radius of sphere
        coords = coords * r

        # Return coordinates plus and minus center
        return center + coords, center - coords

    # Make superordinate centroids
    superCentroids = _centroids_maker(
        center=np.zeros((nFeatures,), dtype=np.float32), r=super_rad
    )

    # Make basic centroids
    basicCentroids = np.zeros((4, nFeatures), dtype=np.float32)
    for i, center in enumerate(superCentroids):
        basicCentroids[(i * 2) : (i * 2 + 2)] = _centroids_maker(
            center=center, r=basic_rad
        )

    # Make subordinate centroids
    subCentroids = np.zeros((8, nFeatures), dtype=np.float32)
    for i, center in enumerate(basicCentroids):
        subCentroids[(i * 2) : (i * 2 + 2)] = _centroids_maker(center=center, r=sub_rad)

    # Generate exemplars
    subExemplars = np.zeros((nImages * 8, nFeatures), dtype=np.float32)
    subLabels = np.zeros((nImages * 8,), dtype=np.int32)
    for i, center in enumerate(subCentroids):
        subExemplars[(i * nImages) : (i * nImages + nImages)] = exemplar_maker(
            nImages,
            center=center,
            radius=cat_rad,
            radius_density=radius_density,
            relu=relu,
        )
        subLabels[(i * nImages) : (i * nImages + nImages)] = i

    return subExemplars, subCentroids, subLabels


class diana:
    def __init__(self, data, metric, max_clusters=None, verbose=False):
        self.data = data
        self.metric = metric
        indices = np.arange(data.shape[0])
        self.tree = Tree()
        self.verbose = verbose

        self.tree.create_node(
            "root",
            0,
            data={
                "indices": indices,
            },
        )

        if max_clusters is None:
            max_clusters = data.shape[0]

        while len(self.tree.leaves()) < max_clusters:
            if self.verbose:
                print(
                    f"We have {len(self.tree.leaves())} clusters, running diana step..."
                )
            # Pick cluster with largest diameter
            nid = self.pick_cluster().identifier

            # Split cluster
            self.split_cluster(nid)

    def _mean_diss(self, simMatrix):
        return np.sum(simMatrix, axis=0) / (simMatrix.shape[0] - 1)

    def split_cluster(self, nid):
        node = self.tree.get_node(nid)

        oldCluster = np.copy(node.data["indices"])
        clusterSim = squareform(pdist(self.data[oldCluster,], metric=self.metric))

        # Find the item that is most dissimilar to the rest of the cluster
        mostDissIdx = np.argmax(self._mean_diss(clusterSim))
        newCluster = oldCluster[mostDissIdx]

        # Remove most dissimilar index from old cluster
        oldCluster = np.delete(oldCluster, mostDissIdx)

        while len(oldCluster) > 1:
            # Compute dissimilarity of old cluster
            oldDiss = squareform(pdist(self.data[oldCluster,], metric=self.metric))
            oldDiss = self._mean_diss(oldDiss)

            # Now compute similarity of each item in the old cluster with the new cluster
            oldClusterData = self.data[oldCluster, :]
            newClusterData = self.data[newCluster, :]

            # if new cluster data is 1D, reshape to 2D
            if len(newClusterData.shape) == 1:
                newClusterData = newClusterData.reshape(1, -1)

            newDiss = (
                np.sum(
                    cdist(oldClusterData, newClusterData, metric=self.metric),
                    axis=1,
                )
                / newClusterData.shape[0]
            )

            # Find new item to remove from old cluster
            dissDiff = oldDiss - newDiss
            mostDissIdx = np.argmax(dissDiff)

            # Check if most dissimilar item is more dissimilar than the new cluster
            if dissDiff[mostDissIdx] < 0:
                break

            # Update clusters
            newCluster = np.append(newCluster, oldCluster[mostDissIdx])
            oldCluster = np.delete(oldCluster, mostDissIdx)

        # Figure out level
        level = self.tree.level(nid) + 1

        # Figure out how many nodes are at this level
        nodesAtLevel = len(
            [
                node
                for node in self.tree.all_nodes()
                if self.tree.level(node.identifier) == level
            ]
        )

        # Figure out the highest nid
        highestNid = np.max([node.identifier for node in self.tree.all_nodes()])

        self.tree.create_node(
            f"level{level}.{nodesAtLevel}",
            highestNid + 1,
            parent=nid,
            data={
                "indices": oldCluster,
            },
        )

        # If new cluster is only 1 element, make it an array
        if not isinstance(newCluster, np.ndarray):
            newCluster = np.array([newCluster])

        self.tree.create_node(
            f"level{level}.{nodesAtLevel + 1}",
            highestNid + 2,
            parent=nid,
            data={
                "indices": newCluster,
            },
        )

        if self.verbose:
            print(
                f"Split cluster {nid} into {highestNid + 1} and {highestNid + 2} at level {level}"
            )
            print(f"Cluster {highestNid + 1} has {len(oldCluster)} objects")
            print(f"Cluster {highestNid + 2} has {len(newCluster)} objects")

    def pick_cluster(self):
        # Get every leaf
        leaves = self.tree.leaves()

        # Calculate diameter of each leaf
        diameters = np.zeros(len(leaves))
        for i, leaf in enumerate(leaves):
            leafData = self.data[leaf.data["indices"], :]
            if len(leafData) == 1:
                diameters[i] = 0
            else:
                diameters[i] = np.max(pdist(leafData, metric=self.metric))

        # Pick the leaf with the largest diameter
        return leaves[np.argmax(diameters)]

    def prune_tree(self, level):
        level += 1
        # Loop through all nodes and delete nodes just after the target level
        for node in self.tree.all_nodes():
            if (
                self.tree.get_node(node.identifier) is not None
                and self.tree.level(node.identifier) == level
            ):
                self.tree.remove_node(node.identifier)

    def linkage_matrix(self, calc_dist=False):
        # Copy tree
        tree = Tree(self.tree.subtree(self.tree.root), deep=True)

        nData = self.data.shape[0]
        # Start building linkage matrix
        linkage = np.zeros((nData - 1, 4))
        rowCount = 0

        # Loop through leaves
        for leaf in tree.leaves():
            # Each leaf is its own cluster, so stick together every object into a bigger and bigger cluster
            cluster = leaf.data["indices"]

            # If the cluster is only one object, just give it a nodeID of itself
            if len(cluster) == 1:
                leaf.data["linkID"] = cluster[0]
                continue

            # Calculate the average distance between objects in the cluster
            if calc_dist:
                clusterReps = self.data[cluster, :]
                clusterDist = np.mean(pdist(clusterReps, metric=self.metric))
            else:
                clusterDist = 0.2

            # Stick the first two items together into a new cluster
            linkage[rowCount, 0] = cluster[0]
            linkage[rowCount, 1] = cluster[1]
            linkage[rowCount, 2] = clusterDist
            linkage[rowCount, 3] = 2
            rowCount += 1

            # Loop through the remaining items and stick it to this cluster
            for i in range(2, len(cluster)):
                linkID = rowCount + nData - 1
                linkage[rowCount, 0] = cluster[i]
                linkage[rowCount, 1] = linkID
                linkage[rowCount, 2] = clusterDist
                linkage[rowCount, 3] = linkage[rowCount - 1, 3] + 1
                rowCount += 1

            # Remember the linkID for this leaf cluster
            leaf.data["linkID"] = rowCount + nData - 1

        # Now loop through the tree and build the rest of the linkage matrix
        for i in range(len(tree.nodes) - 1, -1, -1):
            if i == 0:
                continue

            # if tree.get_node(i) is None:
            #     continue

            # Get the node's parent
            ancestor = tree.get_node(tree.ancestor(i))

            # Only work on this node if the parent doesn't have a linkID yet
            if not "linkID" in ancestor.data:
                # Get the node
                node1 = tree.get_node(i)

                # Get the node's sibling
                node2 = tree.siblings(i)[0]

                # # If sibling doesn't have link ID, skip this for now
                # if not "linkID" in node2.data:
                #     continue

                # Calculate the mean distance bewteen the objects in each node
                if calc_dist:
                    node1Reps = self.data[node1.data["indices"]]
                    node2Reps = self.data[node2.data["indices"]]
                    nodeDist = np.mean(cdist(node1Reps, node2Reps, self.metric))
                else:
                    nodeDist = tree.depth() - tree.level(i) + 1

                # Add the new entry to linkage
                linkID = rowCount + nData
                linkage[rowCount, 0] = node1.data["linkID"]
                linkage[rowCount, 1] = node2.data["linkID"]
                linkage[rowCount, 2] = nodeDist
                linkage[rowCount, 3] = len(ancestor.data["indices"])
                rowCount += 1

                # Save the linkID to the ancestor
                ancestor.data["linkID"] = linkID

        return linkage


class myIPDDP(IPDDP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def linkage_matrix(self, dist_metric=None):
        # Copy tree
        tree = Tree(self.tree.subtree(self.tree.root), deep=True)

        nData = self.samples_number
        # Start building linkage matrix
        linkage = np.zeros((nData - 1, 4))
        rowCount = 0

        # Loop through leaves
        for leaf in tree.leaves():
            # Each leaf is its own cluster, so stick together every object into a bigger and bigger cluster
            cluster = leaf.data["indices"]

            # If the cluster is only one object, just give it a nodeID of itself
            if len(cluster) == 1:
                leaf.data["linkID"] = cluster[0]
                continue

            # Calculate the average distance between objects in the cluster
            if dist_metric is not None:
                clusterReps = self.X[cluster, :]
                clusterDist = np.mean(pdist(clusterReps, metric=dist_metric))
            else:
                clusterDist = 0.2

            # Stick the first two items together into a new cluster
            linkage[rowCount, 0] = cluster[0]
            linkage[rowCount, 1] = cluster[1]
            linkage[rowCount, 2] = clusterDist
            linkage[rowCount, 3] = 2
            rowCount += 1

            # Loop through the remaining items and stick it to this cluster
            for i in range(2, len(cluster)):
                linkID = rowCount + nData - 1
                linkage[rowCount, 0] = cluster[i]
                linkage[rowCount, 1] = linkID
                linkage[rowCount, 2] = clusterDist
                linkage[rowCount, 3] = linkage[rowCount - 1, 3] + 1
                rowCount += 1

            # Remember the linkID for this leaf cluster
            leaf.data["linkID"] = rowCount + nData - 1

        # Now loop through the tree and build the rest of the linkage matrix
        for i in range(len(tree.nodes) - 1, -1, -1):
            if i == 0:
                continue

            # if tree.get_node(i) is None:
            #     continue

            # Get the node's parent
            ancestor = tree.get_node(tree.ancestor(i))

            # Only work on this node if the parent doesn't have a linkID yet
            if not "linkID" in ancestor.data:
                # Get the node
                node1 = tree.get_node(i)

                # Get the node's sibling
                node2 = tree.siblings(i)[0]

                # # If sibling doesn't have link ID, skip this for now
                # if not "linkID" in node2.data:
                #     continue

                # Calculate the mean distance bewteen the objects in each node
                if dist_metric is not None:
                    node1Reps = self.X[node1.data["indices"]]
                    node2Reps = self.X[node2.data["indices"]]
                    nodeDist = np.mean(cdist(node1Reps, node2Reps, metric=dist_metric))
                else:
                    nodeDist = tree.depth() - tree.level(i) + 1

                # Add the new entry to linkage
                linkID = rowCount + nData
                linkage[rowCount, 0] = node1.data["linkID"]
                linkage[rowCount, 1] = node2.data["linkID"]
                linkage[rowCount, 2] = nodeDist
                linkage[rowCount, 3] = len(ancestor.data["indices"])
                rowCount += 1

                # Save the linkID to the ancestor
                ancestor.data["linkID"] = linkID

        return linkage


def get_nodes_at_level(tree, level):
    """Return a list of nodes at a given level of the tree"""
    return [
        i.identifier for i in tree.all_nodes_itr() if tree.level(i.identifier) == level
    ]


def get_leaves_from_node(tree, node):
    """Return the indices of the items (leaves) from a given node"""
    leafList = [leaf.data["indices"] for leaf in tree.leaves(node)]
    return np.concatenate(leafList)


def external_evaluate_over_levels(tree, labels, metric, verbose=False):
    levels = range(1, labels.shape[1] + 1)
    nLeaves = len(tree.get_node(0).data["indices"])

    scores = np.zeros(len(levels))
    for i, level in enumerate(levels):
        levelLabels = labels[:, level - 1]
        nodes = get_nodes_at_level(tree, level)
        levelPred = np.repeat(-1, nLeaves)
        for j, node in enumerate(nodes):
            leaves = get_leaves_from_node(tree, node)

            levelPred[leaves] = j

        # Calculate external metric
        score = metric(levelLabels, levelPred)
        scores[i] = score

        if verbose:
            print(f"Level {level}: {score}")

    return scores


def internal_evaluate_over_levels(tree, reps, metric, level=None, verbose=False):
    if level is None:
        maxLevel = max([tree.level(i.identifier) for i in tree.all_nodes_itr()]) + 1
    else:
        maxLevel = level + 1
    levels = range(1, maxLevel)
    nLeaves = len(tree.get_node(0).data["indices"])

    scores = np.zeros(len(levels))
    for i, level in enumerate(levels):
        nodes = get_nodes_at_level(tree, level)
        levelPred = np.repeat(-1, nLeaves)
        for j, node in enumerate(nodes):
            leaves = get_leaves_from_node(tree, node)

            levelPred[leaves] = j

        # Calculate internal metric
        score = metric(reps, levelPred)
        scores[i] = score

        if verbose:
            print(f"Level {level}: {score}")

    return scores


def calc_cue_validity(exemplars, labels, binary=True, verbose=False):
    categories = np.unique(labels)

    cueValidities = {}
    for category in categories:
        cueValidity = 0
        for k in range(exemplars.shape[1]):
            if binary:
                hasFeature = exemplars[:, k] > 0

                # Check how many images with this feature are in this category
                nImages = np.sum(hasFeature[labels == category])

                cueValidity += nImages / exemplars.shape[0]
            else:
                # Binarize label
                catLabels = np.float32(labels == category)

                # Get features
                featureStrength = exemplars[:, k]

                # Calculate point biserial correlation with fisher Z transform
                cueValidity += np.abs(
                    np.arctanh(np.corrcoef(featureStrength, catLabels)[0, 1])
                )

        cueValidities[category] = cueValidity / exemplars.shape[1]

        if binary:
            # Z to r
            cueValidities[category] = np.tanh(cueValidities[category])

        if verbose:
            print(
                "Category: ",
                category,
                "Cue validity: ",
                np.abs(cueValidity) / exemplars.shape[1],
            )

    return cueValidities


def calc_category_validity(exemplars, labels, binary=True, verbose=False):
    categories = np.unique(labels)

    categoryValidities = {}
    for category in categories:
        categoryImgs = exemplars[category == labels, :]
        category_validity = 0
        for k in range(exemplars.shape[1]):
            if binary:
                # Check how many images has this feature
                hasFeature = np.sum(categoryImgs[:, k] > 0)

                # Add to category_validity
                category_validity += hasFeature / categoryImgs.shape[0]
            else:
                # Average the feature absolute strength
                category_validity += np.mean(np.abs(categoryImgs[:, k]))

        # Save
        categoryValidities[category] = category_validity / exemplars.shape[1]
        if verbose:
            print(
                "Category: ",
                category,
                " Validity: ",
                category_validity / exemplars.shape[1],
            )

    return categoryValidities


def calc_collocation(exemplars, labels, binary=True, verbose=False):
    categories = np.unique(labels)

    collocations = {}
    for category in categories:
        categoryImgs = exemplars[category == labels, :]
        category_validity = 0
        cueValidity = 0
        for k in range(exemplars.shape[1]):
            if binary:
                # Cue validity
                hasFeature = exemplars[:, k] > 0

                # Check how many images with this feature are in this category
                nImages = np.sum(hasFeature[labels == category])

                cueValidity += nImages / exemplars.shape[0]

                # Category validity
                hasFeature = np.sum(categoryImgs[:, k] > 0)

                # Add to category_validity
                category_validity += hasFeature / categoryImgs.shape[0]
            else:
                # Calculate cue validity
                catLabels = np.float32(labels == category)

                # Get features
                features = exemplars[:, k]

                # Calculate correlation
                cueValidity += np.abs(
                    np.arctanh(np.corrcoef(catLabels, features)[0, 1])
                )

                # Calculate category validity
                category_validity += np.mean(np.abs(categoryImgs[:, k]))

        # Divide by number of features
        category_validity /= exemplars.shape[1]
        cueValidity /= exemplars.shape[1]

        collocations[category] = category_validity * cueValidity

        if verbose:
            print(
                "Category: ",
                category,
                " Collocation: ",
                category_validity * cueValidity,
            )

    return collocations


def calc_category_utility(exemplars, labels, binary=True, verbose=False):
    categories = np.unique(labels)

    category_utilities = {}
    for category in categories:
        # Calculate the frequency of this category amongst all labels
        category_frequency = np.sum(labels == category) / labels.shape[0]

        # Loop through features
        category_utility = 0
        for k in range(exemplars.shape[1]):
            categoryImgs = exemplars[category == labels, :]
            if binary:
                # Calculate the frequency that an image has this feature
                feature_frequency = np.sum(exemplars[:, k] > 0) / exemplars.shape[0]

                # Calculate category validity
                hasFeature = np.sum(categoryImgs[:, k] > 0)
                category_validity = hasFeature / categoryImgs.shape[0]

                # Add to category validity
                category_utility += (category_validity**2) - (feature_frequency**2)
            else:
                # Calculate average feature strength (regardless of category)
                feature_strength = np.mean(np.abs(exemplars[:, k]))

                # Calculate category validity
                category_validity = np.mean(np.abs(categoryImgs[:, k]))

                # Category utility
                category_utility += category_validity - feature_strength

        # Multiply by category frequency
        category_utilities[category] = category_utility * category_frequency

        if verbose:
            print(
                "Category: ",
                category,
                " Utility: ",
                category_utility * category_frequency,
            )

    return category_utilities


def print_cluster_stats(tree, hierLabels, exemplars):
    """
    HierLabels must be formatted nxk where k are the levels of the hierarchy
    """
    # Adjusted rand score, 1 is perfect, pair counting method
    print("Adjusted Rand score:")
    _ = external_evaluate_over_levels(
        tree, hierLabels, metrics.adjusted_rand_score, verbose=True
    )

    # Mutual information, 1 is perfect, agreement between two partitions
    print("Adjusted Mutual information:")
    _ = external_evaluate_over_levels(
        tree, hierLabels, metrics.adjusted_mutual_info_score, verbose=True
    )

    # V-measure, 1 is perfect, weighted harmonic mean of homogeneity (cluster only includes one class) and completeness (all members in one class)
    print("V-measure:")
    _ = external_evaluate_over_levels(
        tree, hierLabels, metrics.v_measure_score, verbose=True
    )

    # Fowlkes-Mallows, 1 is perfect, geometric mean between precision (TP/TP+FP) and recall (TP/FP+FN)
    print("Fowlkes-Mallows:")
    _ = external_evaluate_over_levels(
        tree, hierLabels, metrics.fowlkes_mallows_score, verbose=True
    )

    # Davies-Bouldin, 0 is best partitioning, signifies average similarity between clusters
    print("Davies-Bouldin:")
    _ = internal_evaluate_over_levels(
        tree,
        exemplars,
        metrics.davies_bouldin_score,
        level=3,
        verbose=True,
    )

    # Silhouette score, 0 is overlapping clusters, +1 is perfect clustering, -1 is wrong clustering
    print("Silhouette score:")
    _ = internal_evaluate_over_levels(
        tree,
        exemplars,
        metrics.silhouette_score,
        level=3,
        verbose=True,
    )

    # Calinski_harabasz, higher is denser well-separated clusters
    print("Calinski-Harabasz:")
    _ = internal_evaluate_over_levels(
        tree,
        exemplars,
        metrics.calinski_harabasz_score,
        level=3,
        verbose=True,
    )

    return None


def print_category_metrics(exemplars, labels, simMat, imgInfo, binary=True):
    """
    Labels must be formatted nx3 where the second dimension is super, basic, and sub
    """
    superLabels = labels[:, 0]
    basicLabels = labels[:, 1]
    subLabels = labels[:, 2]

    print("Category cue validity = Sum(P(C|fk)) / n")
    print("Superordinate: ")
    calc_cue_validity(exemplars, superLabels, binary=binary, verbose=True)

    print("Basic: ")
    calc_cue_validity(exemplars, basicLabels, binary=binary, verbose=True)

    print("Subordinate: ")
    _ = calc_cue_validity(exemplars, subLabels, binary=binary, verbose=True)
    print()

    print("Category validity = Sum(P(fk|C)) / n")
    print("Superordinate: ")
    calc_category_validity(exemplars, superLabels, binary=binary, verbose=True)

    print("Basic: ")
    calc_category_validity(exemplars, basicLabels, binary=binary, verbose=True)

    print("Subordinate: ")
    _ = calc_category_validity(exemplars, subLabels, binary=binary, verbose=True)
    print()

    print("Collocation (cue validity * category validity)")
    print("Superordinate: ")
    calc_collocation(exemplars, superLabels, binary=binary, verbose=True)

    print("Basic: ")
    calc_collocation(exemplars, basicLabels, binary=binary, verbose=True)

    print("Subordinate: ")
    _ = calc_collocation(exemplars, subLabels, binary=binary, verbose=True)
    print()

    print("Category utility P(C) * Sum(P(Fk|C) ** 2 - P(Fk)**2)")
    print("Superordinate: ")
    calc_category_utility(exemplars, superLabels, binary=binary, verbose=True)

    print("Basic: ")
    calc_category_utility(exemplars, basicLabels, binary=binary, verbose=True)

    print("Subordinate: ")
    _ = calc_category_utility(exemplars, subLabels, binary=binary, verbose=True)
    print()

    print("Cluster index (mean within sim - mean betwen sim)")
    simCluster = SimCluster(simMat=simMat, imgInfo=imgInfo)

    print("Superordinate: ")
    print(simCluster.calculate_index(level="super"))

    print("Basic: ")
    print(simCluster.calculate_index(level="basic"))

    print("Subordinate: ")
    print(simCluster.calculate_index(level="sub"))


def npARI(labels_true, labels_pred, noise_label=-1):
    "Return noise penalized adjusted rand index."
    # Remove the samples that were labeled noise
    noNoiseIdxs = labels_pred != noise_label

    # Calculate ARI for labels not labeled noise
    ari = metrics.adjusted_rand_score(
        labels_true[noNoiseIdxs], labels_pred[noNoiseIdxs]
    )

    # Calculate penalty for noise labels
    noisePenal = (
        labels_true.shape[0] - np.sum(noNoiseIdxs == False)
    ) / labels_true.shape[0]

    return ari * noisePenal


def cartesian_to_polar(coords):
    """
    Convert cartesian coordinates to polar coordinates
    """
    r = np.linalg.norm(coords)
    thetas = np.zeros(len(coords) - 1)

    for i in range(len(thetas)):
        thetas[i] = np.arctan2(np.linalg.norm(coords[i + 1 :]), coords[i])

    return r, thetas


def polar_to_cartesian(r, thetas):
    """
    Convert polar coordinates to cartesian coordinates
    """
    coords = np.zeros(len(thetas) + 1)

    for i in range(len(thetas)):
        coords[i] = r * np.prod(np.sin(thetas[:i])) * np.cos(thetas[i])

    coords[-1] = r * np.prod(np.sin(thetas))

    return coords


class EBRW:
    def __init__(
        self,
        memory_reps: np.ndarray,
        memory_categories: np.ndarray,
        rng: np.random.Generator,
        memory_strengths: np.ndarray = None,
        memory_strength_multiplier: float = 1.0,
        p: float = 2.0,
        c: float = 1.0,
        b: float = 0,
        A: float = 10,
        B: float = 10,
        alpha: float = 1,
    ):
        """
        Return an instance of the EBRW model starting with the given memory
        represntations and their categories (index labels).
        """
        # Memory
        self.memory_reps = memory_reps
        self.memory_categories = memory_categories
        self.memory_strengths = (
            memory_strengths
            if memory_strengths is not None
            else np.ones(len(memory_categories)) / len(memory_categories)
        )
        self.memory_strengths *= memory_strength_multiplier
        self.categories = np.unique(memory_categories)

        if len(self.categories) > 2:
            raise ValueError("We only supports binary categorization")

        self.rng = rng

        # Model parameters
        self.p = p  # Distance metric
        self.c = c  # Sensitivity
        self.b = b  # Criterion/background
        self.A = A  # Category 0 threshold
        self.B = B  # Category 1 threshold
        self.alpha = alpha  # Step time constant

    def _sim(self, probes: np.ndarray, category: int, metric="minkowski", **kwargs):
        """
        Calculates the similarity between the probe items and the
        represenations in category. Calculates distance using cdist with
        defaults for EBRW. Chiefly, default Minkowski distance with p=2 and
        w=1/n_features. The metric and kwargs can be changed to modify this. The
        distance is then used to calculate similarity alongside the c parameter.
        """
        if metric == "minkowski":
            if "p" not in kwargs.keys():
                kwargs["p"] = self.p

            if "w" not in kwargs.keys():
                kwargs["w"] = np.ones(probes.shape[1]) / probes.shape[1]

        # Get memory_reps that match the category
        category_reps = self.memory_reps[self.memory_categories == category]

        if len(category_reps) == 0:
            return np.zeros((probes.shape[0],))

        # Calculate distances
        dists = cdist(probes, category_reps, metric=metric, **kwargs)

        # Calculate similarity
        return (
            np.exp(-self.c * dists)
            * self.memory_strengths[self.memory_categories == category]
        )

    def _sum_sims(self, probes, **kwargs):
        """
        Return the sum similarities given the probes.
        """
        # Calculate sum similarities
        sumA = np.sum(self._sim(probes, self.categories[0], **kwargs), axis=1)

        if len(self.categories) == 1:
            return sumA, 0
        else:
            sumB = np.sum(self._sim(probes, self.categories[1], **kwargs), axis=1)

            return sumA, sumB

    def _prob_step(self, sumA, sumB):
        """
        Return the probability of stepping towards the category from the sum
        similarity.
        """
        p = (sumA + self.b) / (sumA + sumB + (self.b * 2))

        return p, 1 - p

    def categorize(self, probes, categories, **kwargs):
        """
        Return the responses and RT for each probe targetting each category.
        """
        # First calculate sum similarities
        sumA, sumB = self._sum_sims(probes, **kwargs)

        # Calculate step probabilities
        p, q = self._prob_step(sumA, sumB)

        # Calculate probability of category choices
        top = 1 - ((q / p) ** self.B)
        bot = 1 - ((q / p) ** (self.A + self.B))
        pA = top / bot

        top = ((q / p) ** self.B) - ((q / p) ** (self.A + self.B))
        bot = 1 - ((q / p) ** (self.A + self.B))
        pB = top / bot

        # Calculate expected number of steps for each type of response
        # Calculate steps A
        top = ((p / q) ** (self.A + self.B)) + 1
        bot = ((p / q) ** (self.A + self.B)) - 1
        theta1A = top / bot

        top = ((p / q) ** self.B) + 1
        bot = ((p / q) ** self.B) - 1
        theta2A = top / bot

        top = (theta1A * (self.A + self.B)) - (theta2A * self.B)
        bot = p - q
        stepsA = top / bot

        # Calculate steps B
        top = ((p / q) ** -(self.A + self.B)) + 1
        bot = ((p / q) ** -(self.A + self.B)) - 1
        theta1B = top / bot

        top = ((p / q) ** -self.A) + 1
        bot = ((p / q) ** -self.A) - 1
        theta2B = top / bot

        top = (theta1B * (self.A + self.B)) - (theta2B * self.A)
        bot = q - p
        stepsB = top / bot

        # Stick the steps together
        steps = np.stack([stepsA, stepsB], axis=1)

        # Calculate step time
        stepTime = (self.alpha + 1) / (sumA + sumB)

        # Calculate decisions
        decisions = np.zeros((probes.shape[0],), dtype=np.int32)
        rts = np.zeros((probes.shape[0],))
        for i in range(probes.shape[0]):
            if categories[i] == self.categories[0]:
                decision = self.rng.choice([0, 1], p=[pA[i], 1 - pA[i]]).astype(
                    np.int32
                )
                decisions[i] = decision
                rts[i] = steps[i, decision] * stepTime[i]
            else:
                decision = self.rng.choice([0, 1], p=[pB[i], 1 - pB[i]]).astype(
                    np.int32
                )
                decisions[i] = decision
                rts[i] = steps[i, decision] * stepTime[i]

        return decisions, rts


if __name__ == "__main__":
    import ecoset
    from tensorflow.keras.models import Model

    imgPath = "./images/allBirds/train"

    # list files
    files = os.listdir(imgPath)

    # Preallocate array for each iamge
    allImages = np.zeros((len(files), 224, 224, 3))

    for i, file in enumerate(files):
        # Load image
        img = Image.open(os.path.join(imgPath, file))

        # Preprocess image
        img = ecoset.preprocess_alexnet(img)

        # Add to array
        allImages[i, :, :, :] = img

    weightPath = f"./models/AlexNet/ecoset_training_seeds_01_to_10/training_seed_01/model.ckpt_epoch89"
    model = ecoset.make_alex_net_v2(weights_path=weightPath)

    # Get weights for the classification layer
    weights = model.get_layer("fc8").get_weights()[0]

    # Filter for only the birds (idx 25) then flatten
    birdWeights = np.squeeze(weights[:, :, :, 25])
    birdWeights = np.concatenate(
        [np.tile(weight, (5, 5, 1)) for weight in birdWeights], axis=2
    )

    # Get activations from penulatimate layer
    layer = model.get_layer("fc7")

    # Change activation to linear
    layer.activation = tf.keras.activations.linear

    model = Model(inputs=model.inputs, outputs=layer.output)

    # Predict with model with cpu
    with tf.device("/cpu:0"):
        reps = model.predict(allImages)

    # Apply global average pooling
    reps = np.mean(reps, axis=(1, 2))
    categories = np.array([0] * len(reps))

    # Load up test images
    imgPath = "./images/allBirds/test"

    # list files
    files = os.listdir(imgPath)

    # Preallocate array for each iamge
    allImages = np.zeros((len(files), 224, 224, 3))

    for i, file in enumerate(files):
        # Load image
        img = Image.open(os.path.join(imgPath, file))

        # Preprocess image
        img = ecoset.preprocess_alexnet(img)

        # Add to array
        allImages[i, :, :, :] = img

    # Predict with model with cpu
    with tf.device("/cpu:0"):
        probes = model.predict(allImages)

    # Apply global average pooling
    probes = np.mean(probes, axis=(1, 2))

    model = EBRW(memory_reps=reps, memory_categories=categories, b=1000)
    model.categorize(probes, np.array([0] * len(probes)))
