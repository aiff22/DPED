# python test_model.py model=iphone_orig dped_dir=dped/ test_subset=full iteration=all resolution=orig use_gpu=true

from scipy import misc
import numpy as np
import tensorflow as tf
from models import resnet
import utils
import os
import sys

# process command arguments
phone, dped_dir, test_subset, iteration, resolution, use_gpu = utils.process_test_model_args(sys.argv)

# get all available image resolutions
res_sizes = utils.get_resolutions()

# get the specified image resolution
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE = utils.get_specified_res(res_sizes, phone, resolution)

# disable gpu if specified
config = tf.ConfigProto(device_count={'GPU': 0}) if use_gpu == "false" else None

# create placeholders for input images
x_ = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
x_image = tf.reshape(x_, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

# generate enhanced image
enhanced = resnet(x_image)

with tf.Session(config=config) as sess:

    test_dir = dped_dir + phone.replace("_orig", "") + "/test_data/full_size_test_images/"
    test_photos = [f for f in os.listdir(test_dir) if os.path.isfile(test_dir + f)]

    if test_subset == "small":
        # use five first images only
        test_photos = test_photos[0:5]

    if phone.endswith("_orig"):

        # load pre-trained model
        saver = tf.train.Saver()
        saver.restore(sess, "models_orig/" + phone)

        for photo in test_photos:

            # load training image and crop it if necessary

            print("Testing original " + phone.replace("_orig", "") + " model, processing image " + photo)
            image = np.float16(misc.imresize(misc.imread(test_dir + photo), res_sizes[phone])) / 255

            image_crop = utils.extract_crop(image, resolution, phone, res_sizes)
            image_crop_2d = np.reshape(image_crop, [1, IMAGE_SIZE])

            # get enhanced image

            enhanced_2d = sess.run(enhanced, feed_dict={x_: image_crop_2d})
            enhanced_image = np.reshape(enhanced_2d, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

            before_after = np.hstack((image_crop, enhanced_image))
            photo_name = photo.rsplit(".", 1)[0]

            # save the results as .png images

            misc.imsave("visual_results/" + phone + "_" + photo_name + "_enhanced.png", enhanced_image)
            misc.imsave("visual_results/" + phone + "_" + photo_name + "_before_after.png", before_after)

    else:

        num_saved_models = int(len([f for f in os.listdir("models/") if f.startswith(phone + "_iteration")]) / 2)

        if iteration == "all":
            iteration = np.arange(1, num_saved_models) * 1000
        else:
            iteration = [int(iteration)]

        for i in iteration:

            # load pre-trained model
            saver = tf.train.Saver()
            saver.restore(sess, "models/" + phone + "_iteration_" + str(i) + ".ckpt")

            for photo in test_photos:

                # load training image and crop it if necessary

                print("iteration " + str(i) + ", processing image " + photo)
                image = np.float16(misc.imresize(misc.imread(test_dir + photo), res_sizes[phone])) / 255

                image_crop = utils.extract_crop(image, resolution, phone, res_sizes)
                image_crop_2d = np.reshape(image_crop, [1, IMAGE_SIZE])

                # get enhanced image

                enhanced_2d = sess.run(enhanced, feed_dict={x_: image_crop_2d})
                enhanced_image = np.reshape(enhanced_2d, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

                before_after = np.hstack((image_crop, enhanced_image))
                photo_name = photo.rsplit(".", 1)[0]

                # save the results as .png images

                misc.imsave("visual_results/" + phone + "_" + photo_name + "_iteration_" + str(i) + "_enhanced.png", enhanced_image)
                misc.imsave("visual_results/" + phone + "_" + photo_name + "_iteration_" + str(i) + "_before_after.png", before_after)
