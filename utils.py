import scipy.stats as st
import tensorflow as tf
import numpy as np
import sys

from functools import reduce

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter

def blur(x):
    kernel_var = gauss_kernel(21, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')

def process_command_args(arguments):

    # specifying default parameters

    batch_size = 50
    train_size = 30000
    learning_rate = 5e-4
    num_train_iters = 20000

    w_content = 10
    w_color = 0.5
    w_texture = 1
    w_tv = 2000

    dped_dir = 'dped/'
    vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    eval_step = 1000

    phone = ""

    for args in arguments:

        if args.startswith("model"):
            phone = args.split("=")[1]

        if args.startswith("batch_size"):
            batch_size = int(args.split("=")[1])

        if args.startswith("train_size"):
            train_size = int(args.split("=")[1])

        if args.startswith("learning_rate"):
            learning_rate = float(args.split("=")[1])

        if args.startswith("num_train_iters"):
            num_train_iters = int(args.split("=")[1])

        # -----------------------------------

        if args.startswith("w_content"):
            w_content = float(args.split("=")[1])

        if args.startswith("w_color"):
            w_color = float(args.split("=")[1])

        if args.startswith("w_texture"):
            w_texture = float(args.split("=")[1])

        if args.startswith("w_tv"):
            w_tv = float(args.split("=")[1])

        # -----------------------------------

        if args.startswith("dped_dir"):
            dped_dir = args.split("=")[1]

        if args.startswith("vgg_dir"):
            vgg_dir = args.split("=")[1]

        if args.startswith("eval_step"):
            eval_step = int(args.split("=")[1])


    if phone == "":
        print("\nPlease specify the camera model by running the script with the following parameter:\n")
        print("python train_model.py model={iphone,blackberry,sony}\n")
        sys.exit()

    if phone not in ["iphone", "sony", "blackberry"]:
        print("\nPlease specify the correct camera model:\n")
        print("python train_model.py model={iphone,blackberry,sony}\n")
        sys.exit()

    print("\nThe following parameters will be applied for CNN training:\n")

    print("Phone model:", phone)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)
    print("Training iterations:", str(num_train_iters))
    print()
    print("Content loss:", w_content)
    print("Color loss:", w_color)
    print("Texture loss:", w_texture)
    print("Total variation loss:", str(w_tv))
    print()
    print("Path to DPED dataset:", dped_dir)
    print("Path to VGG-19 network:", vgg_dir)
    print("Evaluation step:", str(eval_step))
    print()
    return phone, batch_size, train_size, learning_rate, num_train_iters, \
            w_content, w_color, w_texture, w_tv,\
            dped_dir, vgg_dir, eval_step


def process_test_model_args(arguments):

    phone = ""
    dped_dir = 'dped/'
    test_subset = "small"
    iteration = "all"
    resolution = "orig"
    use_gpu = "true"

    for args in arguments:

        if args.startswith("model"):
            phone = args.split("=")[1]

        if args.startswith("dped_dir"):
            dped_dir = args.split("=")[1]

        if args.startswith("test_subset"):
            test_subset = args.split("=")[1]

        if args.startswith("iteration"):
            iteration = args.split("=")[1]

        if args.startswith("resolution"):
            resolution = args.split("=")[1]

        if args.startswith("use_gpu"):
            use_gpu = args.split("=")[1]

    if phone == "":
        print("\nPlease specify the model by running the script with the following parameter:\n")
        print("python test_model.py model={iphone,blackberry,sony,iphone_orig,blackberry_orig,sony_orig}\n")
        sys.exit()

    return phone, dped_dir, test_subset, iteration, resolution, use_gpu

def get_resolutions():

    # IMAGE_HEIGHT, IMAGE_WIDTH

    res_sizes = {}

    res_sizes["iphone"] = [1536, 2048]
    res_sizes["iphone_orig"] = [1536, 2048]
    res_sizes["blackberry"] = [1560, 2080]
    res_sizes["blackberry_orig"] = [1560, 2080]
    res_sizes["sony"] = [1944, 2592]
    res_sizes["sony_orig"] = [1944, 2592]
    res_sizes["high"] = [1260, 1680]
    res_sizes["medium"] = [1024, 1366]
    res_sizes["small"] = [768, 1024]
    res_sizes["tiny"] = [600, 800]

    return res_sizes

def get_specified_res(res_sizes, phone, resolution):

    if resolution == "orig":
        IMAGE_HEIGHT = res_sizes[phone][0]
        IMAGE_WIDTH = res_sizes[phone][1]
    else:
        IMAGE_HEIGHT = res_sizes[resolution][0]
        IMAGE_WIDTH = res_sizes[resolution][1]

    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3

    return IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE

def extract_crop(image, resolution, phone, res_sizes):

    if resolution == "orig":
        return image

    else:

        x_up = int((res_sizes[phone][1] - res_sizes[resolution][1]) / 2)
        y_up = int((res_sizes[phone][0] - res_sizes[resolution][0]) / 2)

        x_down = x_up + res_sizes[resolution][1]
        y_down = y_up + res_sizes[resolution][0]

        return image[y_up : y_down, x_up : x_down, :]
