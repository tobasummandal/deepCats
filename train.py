import tensorflow as tf
import os
import numpy as np


def create_dataset(directory):
    """
    Create tf.Data.Dataset objects from the train and val directories, assumes
    that we have one directory that has subdirectories so we will have multi-
    class classification.
    """
    # List folders in trainDir
    basicClasses = os.listdir(directory)
    basicClasses.sort()
    nBasic = len(basicClasses)

    # Find the basic class that has subdirectories and count the number
    for folder in basicClasses:
        files = os.listdir(os.path.join(directory, folder))
        if os.path.isdir(os.path.join(directory, folder, files[0])):
            nBirds = len(files)
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
                labels.append((i, nBirds))
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

        # Resize to 128x128
        x = tf.keras.preprocessing.image.smart_resize(x, (128, 128))

        # Center features
        x = 2 * (x / 255 - 0.5)

        # Transpose to channel first format
        x = tf.transpose(x, (2, 0, 1))

        # One-hot encode labels
        y = (tf.one_hot(y[0], nBasic + 1), tf.one_hot(y[1], nBirds + 1))

        return x, y

    ds = (
        tf.data.Dataset.from_generator(
            lambda: zip(imgPaths, labels),
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(2,), dtype=tf.int32),
            ),
        )
        .map(_parse_image)
        .batch(32)
    )

    for batch in ds.take(1):
        print(batch[0].shape)
        print(batch[1])

    return ds


if __name__ == "__main__":
    create_dataset("./images/ecoset/train")
