import os
import PIL.Image as Image
import numpy as np
import tensorflow as tf

# List folders
def build_categories_from_ecoset(
    path, synsets=None, maxImgs=None, includeSub=True
):
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
                                subCat
                                for subCat in subCats
                                if subCat in synsets
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
                            imgs = [
                                img
                                for img in imgs
                                if img.split("_")[0] in synsets
                            ]

                        if maxImgs is not None:
                            imgs = imgs[:maxImgs]

                        cats[name][name2] = [
                            os.path.join(path, name, name2, img)
                            for img in imgs
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


def gcm_sim(rep1, rep2, r=2.0, c=1.0, p=2.0):
    """
    Return the GCM similarity between two representations with equal attention
    weights.
    """
    assert np.all(rep1.shape == rep2.shape)

    weights = np.ones(rep1.shape[0]) / rep1.shape[0]

    dist = np.sum(weights * (np.abs(rep1 - rep2) ** r)) ** (1.0 / r)

    return np.exp(-c * dist ** p)


def sim_prob(rep, cat1Rep, cat2Rep, equalize=False, nExemplars=None):
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
        cat1Rep = cat1Rep[
            np.random.choice(cat1Rep.shape[0], nExemplars, False)
        ]

    if nExemplars is not None and nExemplars < cat2Rep.shape[0]:
        cat2Rep = cat2Rep[
            np.random.choice(cat2Rep.shape[0], nExemplars, False)
        ]

    if equalize:
        if cat1Rep.shape[0] < cat2Rep.shape[0]:
            cat2Rep = cat2Rep[
                np.random.choice(
                    cat2Rep.shape[0], cat1Rep.shape[0], replace=False
                )
            ]
        else:
            cat1Rep = cat1Rep[
                np.random.choice(
                    cat1Rep.shape[0], cat2Rep.shape[0], replace=False
                )
            ]

    rep1 = np.sum(np.apply_along_axis(lambda x: gcm_sim(rep, x), 1, cat1Rep))
    rep2 = np.sum(np.apply_along_axis(lambda x: gcm_sim(rep, x), 1, cat2Rep))

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
