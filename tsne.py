import argparse
from tqdm import tqdm
import cv2, os
import random
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf
import colorsys

from train_veri import VeRi
from train_market1501 import Market1501
from train_mars import Mars
from train_custom import VRAI, Merged_Dataset
from encoder import create_box_encoder, ImageEncoder

from matplotlib.transforms import TransformedBbox, Bbox
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase


###################################
###################################
##                               ##
##       Implementation of       ##
##       the t-SNE plotter       ##
##       for cosine metric       ##
##       learning repo (MM)      ##
##                               ##
###################################
###################################


def fix_random_seeds():
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def make_random_hsv_color():
    h = random.random()
    s = random.random()
    v = random.random()
    return h, s, v

def get_label_color_dict(y_labels):
    labels = np.unique(y_labels)
    label_color_dict={}
    for l in labels:
        h, s, v = make_random_hsv_color()
        c = colorsys.hsv_to_rgb(h, s, v)
        label_color_dict[l] = (c[0]*255, c[1]*255, c[2]*255)
    return label_color_dict

def get_last_index(y, nb_id):
    unique_labels = np.unique(y)
    last_label_to_keep = unique_labels[nb_id]
    y_list = y.tolist()
    last_img_index = y_list.index(last_label_to_keep)

    return last_img_index


def get_features(encoder, dataset, batch_size, num_ids):
    valid_x, valid_y, _ = dataset.read_validation()
    print("Validation set size: %d images, %d identites" % (
        len(valid_x), len(np.unique(valid_y))))
    index = get_last_index(valid_y, num_ids)
    valid_y = valid_y[:index]
    valid_x = valid_x[:index]

    print("sample taken: %d images, %d identites" % (
        len(valid_x), len(np.unique(valid_y))))
    label_color_dict = get_label_color_dict(valid_y)

    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(batch_size)
    features = None
    for d in valid_dataset:
        data_x = d[0].numpy()

        current_features = encoder(data_x, batch_size)
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

    return features, valid_y, valid_x, label_color_dict


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_rectangle_by_class(image, label, label_color_dict):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    color = label_color_dict[label]
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=5)

    return image


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y


def visualize_tsne_images(tx, ty, images, labels, label_color_dict, plot_size=1000, max_image_size=10):
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate

    for image, label, x, y in tqdm(
            zip(images, labels, tx, ty),
            desc='Building the T-SNE plot',
            total=len(images)
    ):

        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)

        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_by_class(image, label, label_color_dict)

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

    plt.imshow(tsne_plot[:, :, ::-1])
    plt.show()

def visualize_tsne_points(tx, ty, labels, label_color_dict):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scs=[]
    # for every class, we'll add a scatter plot separately
    for label in label_color_dict:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([label_color_dict[label][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        s = ax.scatter(current_tx, current_ty, c=color, label=label)
        scs.append(s)

    # build a legend using the labels we set previously
    leg = ax.legend(loc='best', fancybox=True, shadow=True)
    sced = {}
    for legline, origline in zip(leg.get_lines(), scs):
        legline.set_picker(True)  # Enable picking on the legend line.
        sced[legline] = origline


    def on_pick(event):
        # On the pick event, find the original line corresponding to the legend
        # proxy line, and toggle its visibility.
        legline = event.artist
        origline = sced[legline]
        visible = not origline.get_visible()
        origline.set_visible(visible)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled.
        legline.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw()

    # finally, show the plot
    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()


def visualize_tsne(tsne, images, labels, label_color_dict, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points

    print("loading visualisation as colored points")
    visualize_tsne_points(tx, ty, labels, label_color_dict)

    # visualize the plot: samples as images
    print("loading visualisation as images")
    visualize_tsne_images(tx, ty, images, labels, label_color_dict, plot_size=plot_size, max_image_size=max_image_size)


def main():
    tf.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_ids', type=int, default=20)
    parser.add_argument("--model",
        default="output/merged.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--veri_dir", help="Path to VeRi dataset directory.",
        default="data/reid/VeRi")
    parser.add_argument(
        "--market1501_dir", help="Path to market1501 dataset directory.",
        default="data/reid/Market_1501")
    # arg_parser.add_argument(
    #     "--mars_dir", help="Path to mars dataset directory.",
    #     default="resources/VeRi")
    args = parser.parse_args()


    #dataset_market = Market1501(args.market1501_dir, num_validation_y=0.1, seed=1234)

    dataset = VeRi(args.veri_dir, num_validation_y=0.1, seed=1234)

    

    fix_random_seeds()

    encoder = ImageEncoder(args.model)

    # features, labels, images, label_color_dict = get_features(
    #     encoder,
    #     merged_dataset,
    #     batch_size=args.batch_size,
    #     num_images=args.num_images
    # )

    features, labels, images, label_color_dict = get_features(
        encoder,
        dataset,
        batch_size=args.batch_size,
        num_ids=args.num_ids
    )

    tsne = TSNE(n_components=2).fit_transform(features)

    visualize_tsne(tsne, images, labels, label_color_dict)

if __name__ == '__main__':
    main()
