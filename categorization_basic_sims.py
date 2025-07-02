import numpy as np
from scipy.spatial.distance import squareform, cdist
from scipy import stats

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