import numpy as np
import PIL.Image
import tensorflow as tf
import scipy.signal
from image_helper import grayscale_resized_image


# convert kernel matrix to tensor-compatible filter
def make_tf_kernel(k):
    k = np.asarray(k)
    # reshape it to tensorflow 4-D filter
    k = k.reshape(list(k.shape) + [1, 1])
    return tf.constant(k, dtype=tf.float32)


# Simple 2D convolution
def simple_conv2d(x, k):
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
    return y[0, :, :, 0]


def gradient_x(x):
    k = make_tf_kernel([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]])
    return simple_conv2d(x, k)


def gradient_y(x):
    k = make_tf_kernel([[-1, -1, -1],
                        [0, 0, 0],
                        [1, 1, 1]])
    return simple_conv2d(x, k)


# checks whether there exists 7 lines of consistent increasing order in set of lines
def check_match(lineset):
    linediff = np.diff(lineset)
    x = 0
    cnt = 0
    for line in linediff:
        # Within 5 px of the other (allowing for minor image errors)
        if np.abs(line - x) < 5:
            cnt += 1
        else:
            cnt = 0
            x = line
    return cnt == 5


# prunes a set of lines to 7 in consistent increasing order (chessboard)
def prune_lines(lineset):
    linediff = np.diff(lineset)
    x = 0
    cnt = 0
    start_pos = 0
    for i, line in enumerate(linediff):
        # Within 5 px of the other (allowing for minor image errors)
        if np.abs(line - x) < 5:
            cnt += 1
            if cnt == 5:
                end_pos = i + 2
                return lineset[start_pos:end_pos]
        else:
            cnt = 0
            x = line
            start_pos = i
    return lineset


# return skeletonized 1d array (thin to single value, favor to the right)
def skeletonize_1d(arr):
    _arr = arr.copy()  # create a copy of array to modify without destroying original
    # Go forwards
    for i in range(_arr.size - 1):
        # Will right-shift if they are the same
        if arr[i] <= _arr[i + 1]:
            _arr[i] = 0

    # Go reverse
    for i in np.arange(_arr.size - 1, 0, -1):
        if _arr[i - 1] > _arr[i]:
            _arr[i] = 0
    return _arr


# returns pixel indices for the 7 internal chess lines in x and y axes
def get_chess_lines(hdx, hdy, hdx_thresh, hdy_thresh):
    # Blur
    gausswin = scipy.signal.gaussian(21, 4)
    gausswin /= np.sum(gausswin)

    # Blur where there is a strong horizontal or vertical line (binarize)
    blur_x = np.convolve(hdx > hdx_thresh, gausswin, mode='same')
    blur_y = np.convolve(hdy > hdy_thresh, gausswin, mode='same')

    skel_x = skeletonize_1d(blur_x)
    skel_y = skeletonize_1d(blur_y)

    # Find points on skeletonized arrays (where returns 1-length tuple)
    lines_x = np.where(skel_x)[0]  # vertical lines
    lines_y = np.where(skel_y)[0]  # horizontal lines

    # Prune inconsistent lines
    lines_x = prune_lines(lines_x)
    lines_y = prune_lines(lines_y)

    is_match = len(lines_x) == 7 and len(lines_y) == 7 and check_match(lines_x) and check_match(lines_y)

    return lines_x, lines_y, is_match


# Gets a numpy grayscale image and returns lines_x and lines_y
def detect_chessboard(img):
    grey = img

    dX = gradient_x(grey)
    dY = gradient_y(grey)

    dX_pos = tf.clip_by_value(dX, 0., 255., name="dx_positive")
    dX_neg = tf.clip_by_value(dX, -255., 0., name="dx_negative")
    dY_pos = tf.clip_by_value(dY, 0., 255., name="dy_positive")
    dY_neg = tf.clip_by_value(dY, -255., 0., name="dy_negative")

    dX_hough = tf.reduce_sum(dX_pos, 0) * tf.reduce_sum(-dX_neg, 0) / (grey.shape[0] * grey.shape[0])
    dY_hough = tf.reduce_sum(dY_pos, 1) * tf.reduce_sum(-dY_neg, 1) / (grey.shape[1] * grey.shape[1])

    # Arbitrarily choose half of max value as threshold, since they're such strong responses
    dX_hough_thresh = tf.reduce_max(dX_hough) * 0.5
    dY_hough_thresh = tf.reduce_max(dY_hough) * 0.5

    lines_x, lines_y, is_match = get_chess_lines(tf.keras.backend.flatten(dX_hough),
                                                 tf.keras.backend.flatten(dY_hough),
                                                 dX_hough_thresh * .9,
                                                 dY_hough_thresh * .9)

    if is_match:
        print("Chessboard found")
    else:
        print("Couldn't find Chessboard")

    return is_match, lines_x, lines_y


# Split up input grayscale array into 64 tiles stacked in a 3D matrix using the chess linesets
def get_chess_tiles(img, lines_x, lines_y):
    # Find average square size, round to a whole pixel for determining edge pieces sizes
    stepx = np.int32(np.round(np.mean(np.diff(lines_x))))
    stepy = np.int32(np.round(np.mean(np.diff(lines_y))))

    # Pad edges as needed to fill out chessboard (for images that are partially over-cropped)
    #     print stepx, stepy
    #     print "x",lines_x[0] - stepx, "->", lines_x[-1] + stepx, a.shape[1]
    #     print "y", lines_y[0] - stepy, "->", lines_y[-1] + stepy, a.shape[0]
    padr_x = 0
    padl_x = 0
    padr_y = 0
    padl_y = 0

    if lines_x[0] - stepx < 0:
        padl_x = np.abs(lines_x[0] - stepx)
    if lines_x[-1] + stepx > img.shape[1] - 1:
        padr_x = np.abs(lines_x[-1] + stepx - img.shape[1])
    if lines_y[0] - stepy < 0:
        padl_y = np.abs(lines_y[0] - stepy)
    if lines_y[-1] + stepx > img.shape[0] - 1:
        padr_y = np.abs(lines_y[-1] + stepy - img.shape[0])

    # New padded array
    #     print "Padded image to", ((padl_y,padr_y),(padl_x,padr_x))
    a2 = np.pad(img, ((padl_y, padr_y), (padl_x, padr_x)), mode='edge')

    setsx = np.hstack([lines_x[0] - stepx, lines_x, lines_x[-1] + stepx]) + padl_x
    setsy = np.hstack([lines_y[0] - stepy, lines_y, lines_y[-1] + stepy]) + padl_y

    a2 = a2[setsy[0]:setsy[-1], setsx[0]:setsx[-1]]
    setsx -= setsx[0]
    setsy -= setsy[0]
    #     display_array(a2, rng=[0,255])
    #     print "X:",setsx
    #     print "Y:",setsy

    # Matrix to hold images of individual squares (in grayscale)
    #     print "Square size: [%g, %g]" % (stepy, stepx)
    squares = np.zeros([np.round(stepy), np.round(stepx), 64], dtype=np.uint8)

    # For each row
    for i in range(0, 8):
        # For each column
        for j in range(0, 8):
            # Vertical lines
            x1 = setsx[i]
            x2 = setsx[i + 1]
            padr_x = 0
            padl_x = 0
            padr_y = 0
            padl_y = 0

            if (x2 - x1) > stepx:
                if i == 7:
                    x1 = x2 - stepx
                else:
                    x2 = x1 + stepx
            elif (x2 - x1) < stepx:
                if i == 7:
                    # right side, pad right
                    padr_x = stepx - (x2 - x1)
                else:
                    # left side, pad left
                    padl_x = stepx - (x2 - x1)
            # Horizontal lines
            y1 = setsy[j]
            y2 = setsy[j + 1]

            if (y2 - y1) > stepy:
                if j == 7:
                    y1 = y2 - stepy
                else:
                    y2 = y1 + stepy
            elif (y2 - y1) < stepy:
                if j == 7:
                    # right side, pad right
                    padr_y = stepy - (y2 - y1)
                else:
                    # left side, pad left
                    padl_y = stepy - (y2 - y1)
            # slicing a, rows sliced with horizontal lines, cols by vertical lines so reversed
            # Also, change order so its A1,B1...H8 for a white-aligned board
            # Apply padding as defined previously to fit minor pixel offsets
            squares[:, :, (7 - j) * 8 + i] = np.pad(a2[y1:y2, x1:x2], ((padl_y, padr_y), (padl_x, padr_x)), mode='edge')
    return squares
