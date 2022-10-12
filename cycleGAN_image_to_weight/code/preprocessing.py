import tensorflow as tf

""" The preprocessing part is very similar to one done in pix2pix """

BUFFER_SIZE = 175  # 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def resize(image, height, width):
    image = tf.image.resize(image, [height, width],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image


def random_crop(image):
    cropped_image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image


# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)  # casts a tensor to a new type (float32)
    image = (image / 127.5) - 1
    return image


@tf.function()
def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = resize(image, 286, 286)
    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)
    # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image


# If I don't read the image PyCharm doesn't recognize it. The code-line 'tf.io.read_file(image_file) is crucial
def load(image_file):
    # Read and decode an image file to an uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_png(image)
    image = tf.cast(image, tf.float32)

    return image


def preprocess_image_train(image_file):
    image = load(image_file)
    image = random_jitter(image)
    image = normalize(image)

    return image


def preprocess_image_test(image_file):
    image = load(image_file)
    image = resize(image, IMG_WIDTH, IMG_HEIGHT)
    image = normalize(image)

    return image
