import os
import PIL.Image as Image
import numpy as np
import tensorflow as tf
import pandas as pd
from itertools import combinations
from scipy.spatial.distance import pdist, squareform, cdist
from scipy import stats
from treelib import Tree


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
    criterion: float,
    maxImgs: int = None,
) -> pd.DataFrame:
    """
    Simulate a category verification task given a similarity matrix with image
    info using an LBA with a criterion.
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

            # Loop through the test images
            for i, imgSims in enumerate(catSimMat):
                # Randomly select exemplars if needed
                if maxImgs is not None:
                    imgSims = np.random.choice(imgSims, maxImgs, False)

                evidence = np.sum(imgSims)

                # Calculate drift
                drift = evidence / (evidence + criterion)
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
                                "crit": criterion,
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


def default_gcm_sim_mat(reps):
    """
    Calculate a similarity matrix using GCM with r=2, c=1, p=1.
    """
    return np.exp(
        -1
        * squareform(pdist(reps, metric="euclidean"))
        * ((1 / reps.shape[1]) ** (1 / 2))
    )


def exemplar_maker(n, center, radius=1, radius_density="uniform", relu=False):
    nDims = len(center)

    # Generate random numbers as needed
    coords = np.random.normal(loc=0, scale=1, size=(n, nDims))
    uniforms = np.random.uniform(low=0, high=1, size=n)

    if radius_density == "power":
        radii = (uniforms ** (1 / nDims)) * radius
    elif radius_density == "normal":
        radii = (
            (uniforms ** (1 / nDims))
            * np.abs(np.random.normal(loc=0, scale=1, size=n))
            * radius
        )
    elif radius_density == "uniform":
        radii = uniforms * radius
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
    for i, center in enumerate(subCentroids):
        subExemplars[(i * nImages) : (i * nImages + nImages)] = exemplar_maker(
            nImages,
            center=center,
            radius=cat_rad,
            radius_density=radius_density,
            relu=relu,
        )

    return subExemplars, subCentroids


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
                "objects": indices,
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

        oldCluster = np.copy(node.data["objects"])
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
                "objects": oldCluster,
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
                "objects": newCluster,
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
            leafData = self.data[leaf.data["objects"], :]
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
            cluster = leaf.data["objects"]

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
                    node1Reps = self.data[node1.data["objects"]]
                    node2Reps = self.data[node2.data["objects"]]
                    nodeDist = np.mean(cdist(node1Reps, node2Reps, self.metric))
                else:
                    nodeDist = tree.depth() - tree.level(i) + 1

                # Add the new entry to linkage
                linkID = rowCount + nData
                linkage[rowCount, 0] = node1.data["linkID"]
                linkage[rowCount, 1] = node2.data["linkID"]
                linkage[rowCount, 2] = nodeDist
                linkage[rowCount, 3] = len(ancestor.data["objects"])
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
    leafList = [leaf.data["objects"] for leaf in tree.leaves(node)]
    return np.concatenate(leafList)


def external_evaluate_over_levels(tree, labels, metric, verbose=False):
    levels = range(1, labels.shape[1] + 1)
    nLeaves = len(tree.get_node(0).data["objects"])

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
    nLeaves = len(tree.get_node(0).data["objects"])

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


if __name__ == "__main__":

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

    cartOrig = exemplar_maker(1, np.zeros((10,)), radius=1)

    r, pol = cartesian_to_polar(cartOrig[0])
    cartConvert = polar_to_cartesian(r, pol)

    print(cartOrig)
    print(r, pol)
    print(cartConvert)
    print(cartOrig - cartConvert)
