# dataClassifier.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# This file contains feature extraction methods and harness
# code for data classification

from __future__ import division

import mostFrequent
import naiveBayes
import perceptron
import perceptron_pacman
import mira
import samples
import sys
import util
from pacman import GameState

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


# Author: Jean KOSSAIFI <jean.kossaifi@gmail.com>

# import numpy as np
# from numpy import arctan2, fliplr, flipud

def gradient(image, same_size=False):
    """ Computes the Gradients of the image separated pixel difference

    Gradient of X is computed using the filter
        [-1, 0, 1]
    Gradient of X is computed using the filter
        [[1,
          0,
          -1]]

    Parameters
    ----------
    image: image of shape (imy, imx)
    same_size: boolean, optional, default is True
        If True, boundaries are duplicated so that the gradients
        has the same size as the original image.
        Otherwise, the gradients will have shape (imy-2, imx-2)

    Returns
    -------
    (Gradient X, Gradient Y), two numpy array with the same shape as image
        (if same_size=True)
    """
    import numpy as np
    sy, sx = image.shape
    if same_size:
        gx = np.zeros(image.shape)
        gx[:, 1:-1] = -image[:, :-2] + image[:, 2:]
        gx[:, 0] = -image[:, 0] + image[:, 1]
        gx[:, -1] = -image[:, -2] + image[:, -1]

        gy = np.zeros(image.shape)
        gy[1:-1, :] = image[:-2, :] - image[2:, :]
        gy[0, :] = image[0, :] - image[1, :]
        gy[-1, :] = image[-2, :] - image[-1, :]

    else:
        gx = np.zeros((sy-2, sx-2))
        gx[:, :] = -image[1:-1, :-2] + image[1:-1, 2:]

        gy = np.zeros((sy-2, sx-2))
        gy[:, :] = image[:-2, 1:-1] - image[2:, 1:-1]

    return gx, gy


def magnitude_orientation(gx, gy):
    """ Computes the magnitude and orientation matrices from the gradients gx gy

    Parameters
    ----------
    gx: gradient following the x axis of the image
    gy: gradient following the y axis of the image

    Returns
    -------
    (magnitude, orientation)

    Warning
    -------
    The orientation is in degree, NOT radian!!
    """
    import numpy as np
    from numpy import arctan2

    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = (arctan2(gy, gx) * 180 / np.pi) % 360

    return magnitude, orientation


def compute_coefs(csx, csy, dx, dy, n_cells_x, n_cells_y):
    """
    Computes the coefficients for the bilinear (spatial) interpolation

    Parameters
    ----------
    csx: int
        number of columns of the cells
    csy: int
        number of raws dimension of the cells
    sx: int
        number of colums of the image (x axis)
    sy: int
        number of raws of the image (y axis)
    n_cells_x: int
        number of cells in the x axis
    n_cells_y: int
        number of cells in the y axis

    Notes
    -----
    We consider an image: image[y, x] (NOT image[x, y]!!!)

    /!\ csx and csy must be even number

    Using the coefficients
    ----------------------
    The coefficient correspond to the interpolation in direction of the upper left corner of the image.
    In other words, if you interpolate img, and res is the result of your interpolation, you should do

    res = zeros(n_cells_y*pixels_per_cell, n_cells_x*pixels_per_cell)
        with (csx, csy) the number of pixels per cell
         and dx, dy = csx//2, csy//2
    res[:-dx, :-dy] += img[dx:, dy:]*coefs

    then you rotate the coefs and do the same thing for every part of the image
    """
    import numpy as np
    from numpy import fliplr, flipud
    if csx != csy:
        raise NotImplementedError("For now compute_coefs is only implemented for squared cells (csx == csy)")

        ################################
        #####     /!\ TODO  /!|    #####
        ################################

    else: # Squared cells
        # Note: in this case, dx = dy, we differentiate them only to make the code clearer

        # We want a squared coefficients matrix so that it can be rotated to interpolate in every direction
        n_cells = max(n_cells_x, n_cells_y)

        # Every cell of this matrix corresponds to (x - x_1)/dx
        x = (np.arange(dx)+0.5)/csx

        # Every cell of this matrix corresponds to (y - y_1)/dy
        y = (np.arange(dy)+0.5)/csy

        y = y[np.newaxis, :]
        x = x[:, np.newaxis]

        # CENTRAL COEFFICIENT
        ccoefs = np.zeros((csy, csx))

        ccoefs[:dy, :dx] = (1 - x)*(1 - y)
        ccoefs[:dy, -dx:] = fliplr(y)*(1 - x)
        ccoefs[-dy:, :dx] = (1 - y)*flipud(x)
        ccoefs[-dy:, -dx:] = fliplr(y)*flipud(x)

        coefs = np.zeros((csx*n_cells - dx, csy*n_cells - dy))
        coefs[:-dy, :-dx] = np.tile(ccoefs, (n_cells - 1, n_cells - 1))

        # REST OF THE BORDER
        coefs[:-dy, -dx:] = np.tile(np.concatenate(((1 - x), np.flipud(x))), (n_cells - 1, dy))
        coefs[-dy:, :-dx] = np.tile(np.concatenate(((1 - y), np.fliplr(y)), axis=1), (dx, n_cells - 1))
        coefs[-dy:, -dx:] = 1

        return coefs


def interpolate_orientation(orientation, sx, sy, nbins, signed_orientation):
    """ interpolates linearly the orientations to their corresponding bins

    Parameters
    ----------
    sx: int
        number of columns of the image (x axis)
    sy: int
        number of raws of the image (y axis)
    nbins : int, optional, default is 9
        Number of orientation bins.
    signed_orientation: bool, default is True
        if True, sign information of the orientation is preserved,
            ie orientation angles are between 0 and 360 degree.
        if False, the angles are between 0 and 180 degree.

    Returns
    -------
    pre-histogram: array of shape (sx, sy, nbins)
            contains the pre histogram of orientation built using linear interpolation
            to interpolate the orientations to their bins
    """
    import numpy as np

    if signed_orientation:
        max_angle = 360
    else:
        max_angle = 180

    b_step = max_angle/nbins
    b0 = (orientation % max_angle) // b_step
    b0[np.where(b0>=nbins)]=0
    b1 = b0 + 1
    b1[np.where(b1>=nbins)]=0
    b = np.abs(orientation % b_step) / b_step

    #linear interpolation between the bins
    # Coefficients corresponding to the bin interpolation
    # We go from an image to a higher dimension representation of size (sizex, sizey, nbins)
    temp_coefs = np.zeros((sy, sx, nbins))
    for i in range(nbins):
        temp_coefs[:, :, i] += np.where(b0==i, (1 - b), 0)
        temp_coefs[:, :, i] += np.where(b1==i, b, 0)

    return temp_coefs


def per_pixel_hog(image, dy=2, dx=2, signed_orientation=False, nbins=9, flatten=False, normalise=True):
    """ builds a histogram of orientation for a cell centered around each pixel of the image

    Parameters
    ---------
    image: numpy array of shape (sizey, sizex)
    dx   : the cell around each pixel in the x axis will have size 2*dx+1
    dy   : the cell around each pixel in the y axis will have size 2*dy+1
    signed_orientation: bool, default is True
        if True, sign information of the orientation is preserved,
            ie orientation angles are between 0 and 360 degree.
        if False, the angles are between 0 and 180 degree.
    nbins : int, optional, default is 9
        Number of orientation bins.

    Returns
    -------
    if visualise if True: (histogram of oriented gradient, visualisation image)

    histogram of oriented gradient:
        numpy array of shape (n_cells_y, n_cells_x, nbins), flattened if flatten is True
    """
    gx, gy = gradient(image, same_size=True)
    magnitude, orientation = magnitude_orientation(gx, gy)
    sy, sx = image.shape
    orientations_image = interpolate_orientation(orientation, sx, sy, nbins, signed_orientation)
    for j in range(1, dy):
        for i in range(1, dx):
            orientations_image[:-j, :-i, :] += orientations_image[j:, i:, :]

    if normalise:
        normalised_blocks = normalise_histogram(orientations_image, 1, 1, 1, 1, nbins)
    else:
        normalised_blocks = orientations_image

    if flatten:
        normalised_blocks = normalised_blocks.flatten()

    return normalised_blocks


def interpolate(magnitude, orientation, csx, csy, sx, sy, n_cells_x, n_cells_y, signed_orientation=False, nbins=9):
    """ Returns a matrix of size (cell_size_x, cell_size_y, nbins) corresponding
         to the trilinear interpolation of the pixels magnitude and orientation

    Parameters
    ----------
    csx: int
        number of columns of the cells
    csy: int
        number of raws dimension of the cells
    sx: int
        number of colums of the image (x axis)
    sy: int
        number of raws of the image (y axis)
    n_cells_x: int
        number of cells in the x axis
    n_cells_y: int
        number of cells in the y axis
    signed_orientation: bool, default is True
        if True, sign information of the orientation is preserved,
            ie orientation angles are between 0 and 360 degree.
        if False, the angles are between 0 and 180 degree.
    nbins : int, optional, default is 9
        Number of orientation bins.

    Returns
    -------
    orientation_histogram: array of shape (n_cells_x, n_cells_y, nbins)
            contains the histogram of orientation built using tri-linear interpolation
    """
    import numpy as np

    dx = csx//2
    dy = csy//2

    temp_coefs = interpolate_orientation(orientation, sx, sy, nbins, signed_orientation)


    # Coefficients of the spatial interpolation in every direction
    coefs = compute_coefs(csx, csy, dx, dy, n_cells_x, n_cells_y)

    temp = np.zeros((sy, sx, nbins))
    # hist(y0, x0)
    temp[:-dy, :-dx, :] += temp_coefs[dy:, dx:, :]*\
        (magnitude[dy:, dx:]*coefs[-(n_cells_y*csy - dy):, -(n_cells_x*csx - dx):])[:, :, np.newaxis]

    # hist(y1, x0)
    coefs = np.rot90(coefs)
    temp[dy:, :-dx, :] += temp_coefs[:-dy, dx:, :]*\
        (magnitude[:-dy, dx:]*coefs[:(n_cells_y*csy - dy), -(n_cells_x*csx - dx):])[:, :, np.newaxis]

    # hist(y1, x1)
    coefs = np.rot90(coefs)
    temp[dy:, dx:, :] += temp_coefs[:-dy, :-dx, :]*\
        (magnitude[:-dy, :-dx]*coefs[:(n_cells_y*csy - dy), :(n_cells_x*csx - dx)])[:, :, np.newaxis]

    # hist(y0, x1)
    coefs = np.rot90(coefs)
    temp[:-dy, dx:, :] += temp_coefs[dy:, :-dx, :]*\
        (magnitude[dy:, :-dx]*coefs[-(n_cells_y*csy - dy):, :(n_cells_x*csx - dx)])[:, :, np.newaxis]

    # Compute the histogram: sum over the cells
    orientation_histogram = temp.reshape((n_cells_y, csy, n_cells_x, csx, nbins)).sum(axis=3).sum(axis=1)

    return orientation_histogram


def draw_histogram(hist, csx, csy, signed_orientation=False):
    """ simple function to draw an orientation histogram
        with arrows
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if signed_orientation:
        max_angle = 2*np.pi
    else:
        max_angle = np.pi

    n_cells_y, n_cells_x, nbins = hist.shape
    sx, sy = n_cells_x*csx, n_cells_y*csy
    plt.close()
    plt.figure()#figsize=(sx/2, sy/2))#, dpi=1)
    plt.xlim(0, sx)
    plt.ylim(sy, 0)
    center = csx//2, csy//2
    b_step = max_angle / nbins

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            for k in range(nbins):
                if hist[i, j, k] != 0:
                    width = 1*hist[i, j, k]
                    plt.arrow((center[0] + j*csx) - np.cos(b_step*k)*(center[0] - 1),
                              (center[1] + i*csy) + np.sin(b_step*k)*(center[1] - 1),
                          2*np.cos(b_step*k)*(center[0] - 1), -2*np.sin(b_step*k)*(center[1] - 1),
                          width=width, color=str(width), #'black',
                          head_width=2.2*width, head_length=2.2*width,
                          length_includes_head=True)

    plt.show()


def visualise_histogram(hist, csx, csy, signed_orientation=False):
    """ Create an image visualisation of the histogram of oriented gradient

    Parameters
    ----------
    hist: numpy array of shape (n_cells_y, n_cells_x, nbins)
        histogram of oriented gradient
    csx: int
        number of columns of the cells
    csy: int
        number of raws dimension of the cells
    signed_orientation: bool, default is True
        if True, sign information of the orientation is preserved,
            ie orientation angles are between 0 and 360 degree.
        if False, the angles are between 0 and 180 degree.

    Return
    ------
    Image of shape (hist.shape[0]*csy, hist.shape[1]*csx)
    """
    import numpy as np
    from skimage import draw

    if signed_orientation:
        max_angle = 2*np.pi
    else:
        max_angle = np.pi

    n_cells_y, n_cells_x, nbins = hist.shape
    sx, sy = n_cells_x*csx, n_cells_y*csy
    center = csx//2, csy//2
    b_step = max_angle / nbins

    radius = min(csx, csy) // 2 - 1
    hog_image = np.zeros((sy, sx), dtype=float)
    for x in range(n_cells_x):
        for y in range(n_cells_y):
            for o in range(nbins):
                centre = tuple([y * csy + csy // 2, x * csx + csx // 2])
                dx = radius * np.cos(o*nbins)
                dy = radius * np.sin(o*nbins)
                rr, cc = draw.line(int(centre[0] - dy),
                                   int(centre[1] - dx),
                                   int(centre[0] + dy),
                                   int(centre[1] + dx))
                hog_image[rr, cc] += hist[y, x, o]
    return hog_image


def normalise_histogram(orientation_histogram, bx, by, n_cells_x, n_cells_y, nbins):
    """ normalises a histogram by blocks

    Parameters
    ----------
    bx: int
        number of blocks on the x axis
    by: int
        number of blocks on the y axis
    n_cells_x: int
        number of cells in the x axis
    n_cells_y: int
        number of cells in the y axis
    nbins : int, optional, default is 9
        Number of orientation bins.

    The normalisation is done according to Dalal's original thesis, using L2-Hys.
    In other words the histogram is first normalised block-wise using l2 norm, before clipping it by
        limiting the values between 0 and 0.02 and finally normalising again with l2 norm
    """
    import numpy as np
    eps = 1e-7

    if bx==1 and by==1: #faster version
        normalised_blocks = np.clip(
          orientation_histogram / np.sqrt(orientation_histogram.sum(axis=-1)**2 + eps)[:, :, np.newaxis], 0, 0.2)
        normalised_blocks /= np.sqrt(normalised_blocks.sum(axis=-1)**2 + eps)[:, :, np.newaxis]

    else:
        n_blocksx = (n_cells_x - bx) + 1
        n_blocksy = (n_cells_y - by) + 1
        normalised_blocks = np.zeros((n_blocksy, n_blocksx, nbins))

        for x in range(n_blocksx):
            for y in range(n_blocksy):
                block = orientation_histogram[y:y + by, x:x + bx, :]
                normalised_blocks[y, x, :] = np.clip(block[0, 0, :] / np.sqrt(block.sum()**2 + eps), 0, 0.2)
                normalised_blocks[y, x, :] /= np.sqrt(normalised_blocks[y, x, :].sum()**2 + eps)

    return normalised_blocks


def build_histogram(magnitude, orientation, cell_size=(8, 8), signed_orientation=False,
         nbins=9, cells_per_block=(1, 1), visualise=False, flatten=False, normalise=True):
    """ builds a histogram of orientation using the provided magnitude and orientation matrices

    Parameters
    ---------
    magnitude: np-array of size (sy, sx)
        matrix of magnitude
    orientation: np-array of size (sy, sx)
        matrix of orientations
    csx: int
        number of columns of the cells
        MUST BE EVEN
    csy: int
        number of raws dimension of the cells
        MUST BE EVEN
    sx: int
        number of colums of the image (x axis)
    sy: int
        number of raws of the image (y axis)
    n_cells_x: int
        number of cells in the x axis
    n_cells_y: int
        number of cells in the y axis
    signed_orientation: bool, default is True
        if True, sign information of the orientation is preserved,
            ie orientation angles are between 0 and 360 degree.
        if False, the angles are between 0 and 180 degree.
    nbins : int, optional, default is 9
        Number of orientation bins.

    Returns
    -------
    if visualise if True: (histogram of oriented gradient, visualisation image)

    histogram of oriented gradient:
        numpy array of shape (n_cells_y, n_cells_x, nbins), flattened if flatten is True
    visualisation image:
        Image of shape (hist.shape[0]*csy, hist.shape[1]*csx)
    """
    sy, sx = magnitude.shape
    csy, csx = cell_size

    # checking that the cell size are even
    if csx % 2 != 0:
        csx += 1
        print("WARNING: the cell_size must be even, incrementing cell_size_x of 1")
    if csy % 2 != 0:
        csy += 1
        print("WARNING: the cell_size must be even, incrementing cell_size_y of 1")

    # Consider only the right part of the image
    # (if the rest doesn't fill a whole cell, just drop it)
    sx -= sx % csx
    sy -= sy % csy
    n_cells_x = sx//csx
    n_cells_y = sy//csy
    magnitude = magnitude[:sy, :sx]
    orientation = orientation[:sy, :sx]
    by, bx = cells_per_block

    orientation_histogram = interpolate(magnitude, orientation, csx, csy, sx, sy, n_cells_x, n_cells_y, signed_orientation, nbins)

    if normalise:
        normalised_blocks = normalise_histogram(orientation_histogram, bx, by, n_cells_x, n_cells_y, nbins)
    else:
        normalised_blocks = orientation_histogram

    if flatten:
        normalised_blocks = normalised_blocks.flatten()

    if visualise:
        #draw_histogram(normalised_blocks, csx, csy, signed_orientation)
        return normalised_blocks, visualise_histogram(normalised_blocks, csx, csy, signed_orientation)
    else:
        return normalised_blocks


def histogram_from_gradients(gradientx, gradienty, cell_size=(8, 8), cells_per_block=(1, 1), signed_orientation=False,
        nbins=9, visualise=False, normalise=True, flatten=False, same_size=False):
    """ builds a histogram of oriented gradient from the provided gradients

    Parameters
    ----------
    gradientx : (M, N) ndarray
        Gradient following the x axis
    gradienty: (M, N) ndarray
        Gradient following the y axis
    nbins : int, optional, default is 9
        Number of orientation bins.
    cell_size : 2 tuple (int, int), optional, default is (8, 8)
        Size (in pixels) of a cell.
    cells_per_block : 2 tuple (int,int), optional, default is (2, 2)
        Number of cells in each block.
    visualise : bool, optional, default is False
        Also return an image of the HOG.
    flatten: bool, optional, default is True
    signed_orientation: bool, default is True
        if True, sign information of the orientation is preserved,
            ie orientation angles are between 0 and 360 degree.
        if False, the angles are between 0 and 180 degree.
    normalise: bool, optional, default is True
        if True, the histogram is normalised block-wise
    same_size: bool, optional, default is False
        if True, the boundaries are duplicated when computing the gradients of the image
        so that these have the same size as the original image

    Returns
    -------
    if visualise if True: (histogram of oriented gradient, visualisation image)

    histogram of oriented gradient:
        numpy array of shape (n_cells_y, n_cells_x, nbins), flattened if flatten is True
    visualisation image:
        Image of shape (hist.shape[0]*csy, hist.shape[1]*csx)

    References
    ----------
    * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

    * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
    Human Detection, IEEE Computer Society Conference on Computer
    Vision and Pattern Recognition 2005 San Diego, CA, USA
    """
    magnitude, orientation = magnitude_orientation(gradientx, gradienty)
    return build_histogram(magnitude, orientation, cell_size=cell_size,
         signed_orientation=signed_orientation, cells_per_block=cells_per_block,
         nbins=nbins, visualise=visualise, normalise=normalise, flatten=flatten)


def hog(image, cell_size=(4, 4), cells_per_block=(1, 1), signed_orientation=False,
        nbins=18, visualise=False, normalise=True, flatten=False, same_size=True):
    """ builds a histogram of oriented gradient (HoG) from the provided image

    Compute a Histogram of Oriented Gradients (HOG) by

    1. computing the gradient image in x and y and deduce from them the magnitude and orientation
        of each pixel
    2. computing gradient histograms (vectorised version)
    3. normalising across blocks
    4. flattening into a feature vector if flatten=True

    Parameters
    ----------
    image : (M, N) ndarray
        Input image (greyscale).
    nbins : int, optional, default is 9
        Number of orientation bins.
    cell_size : 2 tuple (int, int), optional, default is (8, 8)
        Size (in pixels) of a cell.
    cells_per_block : 2 tuple (int,int), optional, default is (2, 2)
        Number of cells in each block.
    visualise : bool, optional, default is False
        Also return an image of the HOG.
    flatten: bool, optional, default is True
    signed_orientation: bool, default is True
        if True, sign information of the orientation is preserved,
            ie orientation angles are between 0 and 360 degree.
        if False, the angles are between 0 and 180 degree.
    normalise: bool, optional, default is True
        if True, the histogram is normalised block-wise
    same_size: bool, optional, default is True
        if True, the boundaries are duplicated when computing the gradients of the image
        so that these have the same size as the original image

    Returns
    -------
    if visualise if True: (histogram of oriented gradient, visualisation image)

    histogram of oriented gradient:
        numpy array of shape (n_cells_y, n_cells_x, nbins), flattened if flatten is True
    visualisation image:
        Image of shape (hist.shape[0]*csy, hist.shape[1]*csx)

    References
    ----------
    * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

    * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
    Human Detection, IEEE Computer Society Conference on Computer
    Vision and Pattern Recognition 2005 San Diego, CA, USA
    """
    gx, gy = gradient(image, same_size=same_size)
    return histogram_from_gradients(gx, gy, cell_size=cell_size,
         signed_orientation=signed_orientation, cells_per_block=cells_per_block,
         nbins=nbins, visualise=visualise, normalise=normalise, flatten=flatten)

def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def basicFeatureExtractorFace(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is an edge (1) or no edge (0)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def enhancedFeatureExtractorDigit(datum):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for this datum (datum is of type samples.Datum).

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ## Something you might need to use:
    ## DIGIT_DATUM_WIDTH : pixel width of digit
    ## DIGIT_DATUM_HEIGHT: pixel height of digit
    ## datum.getPixel(x, y): get pixels of digit
    """
    features =  basicFeatureExtractorDigit(datum)

    "*** YOUR CODE HERE ***"
    import numpy as np
    data = []
    for y in range(DIGIT_DATUM_HEIGHT):
        tmp = []
        for x in range(DIGIT_DATUM_WIDTH):
            tmp.append(datum.getPixel(x, y))
        data.append(np.array(tmp))
    data = np.array(data)
    hog_feature = hog(data, flatten=True)
    for i in range(hog_feature.shape[0]):
        features['hog_feature_' + str(i)] = hog_feature[i]

    visit = util.Counter()
    num_of_components = 0
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if visit[(x, y)] == 0 and datum.getPixel(x, y) == 0:
                num_of_components += 1
                q = util.Queue()
                q.push((x, y))
                visit[(x, y)] = 1
                while not q.isEmpty():
                    nowx, nowy = q.pop()
                    if nowx >= 1 and visit[(nowx - 1, nowy)] == 0 and datum.getPixel(nowx - 1, nowy) == 0:
                        q.push((nowx - 1, nowy))
                        visit[(nowx - 1, nowy)] = 1
                    if nowx <= DIGIT_DATUM_WIDTH - 2 and visit[(nowx + 1, nowy)] == 0 and datum.getPixel(nowx + 1, nowy) == 0:
                        q.push((nowx + 1, nowy))
                        visit[(nowx + 1, nowy)] = 1
                    if nowy >= 1 and visit[(nowx, nowy - 1)] == 0 and datum.getPixel(nowx, nowy - 1) == 0:
                        q.push((nowx, nowy - 1))
                        visit[(nowx, nowy - 1)] = 1
                    if nowy <= DIGIT_DATUM_HEIGHT - 2 and visit[(nowx, nowy + 1)] == 0 and datum.getPixel(nowx, nowy + 1) == 0:
                        q.push((nowx, nowy + 1))
                        visit[(nowx, nowy + 1)] = 1
    num = [0, 0, 0]
    num[min([num_of_components - 1, len(num) - 1])] = 1
    for i in range(len(num)):
        features[str(i)] = num[i]

    num = 0
    total_cut = 1
    for cut in range(1, total_cut + 1):
        for i in range(DIGIT_DATUM_WIDTH):
            num += datum.getPixel(i, int(DIGIT_DATUM_HEIGHT / float(total_cut + 1) * float(cut))) > 0
        for i in range(1, 11):
            features[('cut', 'h', cut, i)] = num > int(DIGIT_DATUM_WIDTH / float(total_cut + 1) * float(cut) * float(i) / 10.0)

    num = 0
    total_cut = 1
    for cut in range(1, total_cut + 1):
        for i in range(DIGIT_DATUM_HEIGHT):
            num += datum.getPixel(int(DIGIT_DATUM_WIDTH / float(total_cut + 1) * float(cut)), i) > 0
        for i in range(1, 11):
            features[('cut', 'v', cut, i)] = num > int(DIGIT_DATUM_WIDTH / float(total_cut + 1) * float(cut) * float(i) / 10.0)

    for y in range(DIGIT_DATUM_HEIGHT):
        num = 0
        for x in range(1, DIGIT_DATUM_WIDTH):
            prv = datum.getPixel(x - 1, y)
            if prv > 0:
                prv = 1
            now = datum.getPixel(x, y)
            if now > 0:
                now = 1
            num += (now + prv) == 1
        features[('h', y, 0)] = 0 <= num < 2
        features[('h', y, 1)] = 2 <= num < 4
        features[('h', y, 2)] = 4 <= num < 6

    for x in range(DIGIT_DATUM_WIDTH):
        num = 0
        for y in range(1, DIGIT_DATUM_HEIGHT):
            prv = datum.getPixel(x, y - 1)
            if prv > 0:
                prv = 1
            now = datum.getPixel(x, y)
            if now > 0:
                now = 1
            num += (now + prv) == 1
        features[('v', x, 0)] = 0 <= num < 2
        features[('v', x, 1)] = 2 <= num < 4
        features[('v', x, 2)] = 4 <= num < 6

    return features
enhancedFeatureExtractorDigit.counter = 0



def basicFeatureExtractorPacman(state):
    """
    A basic feature extraction function.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """
    features = util.Counter()
    for action in state.getLegalActions():
        successor = state.generateSuccessor(0, action)
        foodCount = successor.getFood().count()
        featureCounter = util.Counter()
        featureCounter['foodCount'] = foodCount
        features[action] = featureCounter
    return features, state.getLegalActions()

def enhancedFeatureExtractorPacman(state):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """

    features = basicFeatureExtractorPacman(state)[0]
    for action in state.getLegalActions():
        features[action] = util.Counter(features[action], **enhancedPacmanFeatures(state, action))
    return features, state.getLegalActions()

def get_distance(currentGameState, ghost_set, dist):
    sx, sy = currentGameState.getPacmanPosition()
    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    queue = util.Queue()
    queue.push((sx, sy))
    all_dist = dict()
    all_dist[(sx, sy)] = 0
    while not queue.isEmpty():
        x, y = queue.pop()
        now_dist = all_dist[(x, y)]
        if (x, y) in dist:
            dist[(x, y)] = now_dist
        for direction in directions:
            dx, dy = direction[0], direction[1]
            new_x, new_y = x + dx, y + dy
            if (new_x, new_y) in all_dist:
                continue
            if currentGameState.hasWall(new_x, new_y) or (new_x, new_y) in ghost_set:
                continue
            queue.push((new_x, new_y))
            all_dist[(new_x, new_y)] = now_dist + 1

def get_distance2(currentGameState, dist):
    sx, sy = currentGameState.getPacmanPosition()
    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    queue = util.Queue()
    queue.push((sx, sy))
    all_dist = dict()
    all_dist[(sx, sy)] = 0
    while not queue.isEmpty():
        x, y = queue.pop()
        now_dist = all_dist[(x, y)]
        if (x, y) in dist:
            dist[(x, y)] = now_dist
        for direction in directions:
            dx, dy = direction[0], direction[1]
            new_x, new_y = x + dx, y + dy
            if (new_x, new_y) in all_dist:
                continue
            if currentGameState.hasWall(new_x, new_y):
                continue
            queue.push((new_x, new_y))
            all_dist[(new_x, new_y)] = now_dist + 1

def enhancedPacmanFeatures(state, action):
    """
    For each state, this function is called with each legal action.
    It should return a counter with { <feature name> : <feature value>, ... }

    ## Something you might need to use:
    ## state.generateSuccessor(0, action): state after taking the action (a GameState)
    ## For other information in a GameState
    ## Please refer to class GameState in pacman.py
    """
    features = util.Counter()
    "*** YOUR CODE HERE ***"
    successor = state.generateSuccessor(0, action)
    features['current_score'] = successor.getScore()

    sx, sy = successor.getPacmanPosition()
    dist = dict()
    ghost_list = successor.getGhostPositions()
    ghost_set = set()
    scared_times = [ghost_state.scaredTimer for ghost_state in successor.getGhostStates()]
    food_set = set(successor.getFood().asList())
    capsule_set = set(successor.getCapsules())

    # get real distance of all foods
    for i, (x, y) in enumerate(ghost_list):
        if scared_times[i] < int((abs(sx - x) + abs(sy - y)) / 2.0):
            ghost_set.add((int(x), int(y)))
    for x, y in list(food_set):
        dist[(x, y)] = 1e9
    for x, y in list(capsule_set):
        dist[(x, y)] = 1e9
    get_distance(successor, ghost_set, dist)

    for pos, dist in dist.iteritems():
        if pos in food_set:
            features['food_score'] += 1.0 / float(dist) # 30
        if pos in capsule_set:
            features['capsule_score'] += 100.0 / float(dist)

    for i, (x, y) in enumerate(ghost_list):
        if scared_times[i] > 0:
            features['scared_score'] += 50.0 / float(abs(sx - x) + abs(sy - y) + 1e-8) # 50
            features['scared_score2'] += 40.0 / float(abs(sx - x) + abs(sy - y) + 1e-8) # 50
            features['scared_score3'] += 30.0 / float(abs(sx - x) + abs(sy - y) + 1e-8) # 50

    ghost_dist = dict()
    for pos in ghost_list:
        ghost_dist[pos] = 0.0
    get_distance2(successor, ghost_dist)
    for pos, dist in ghost_dist.iteritems():
        features['ghost_dist'] += float(dist) * 10.0
        features['ghost_dist2'] += float(dist) * 10.0

    return features


def contestFeatureExtractorDigit(datum):
    """
    Specify features to use for the minicontest
    """
    features =  basicFeatureExtractorDigit(datum)
    return features

def enhancedFeatureExtractorFace(datum):
    """
    Your feature extraction playground for faces.
    It is your choice to modify this.
    """
    features =  basicFeatureExtractorFace(datum)
    return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the printImage(<list of pixels>) function to visualize features.

    An example of use has been given to you.

    - classifier is the trained classifier
    - guesses is the list of labels predicted by your classifier on the test set
    - testLabels is the list of true labels
    - testData is the list of training datapoints (as util.Counter of features)
    - rawTestData is the list of training datapoints (as samples.Datum)
    - printImage is a method to visualize the features
    (see its use in the odds ratio part in runClassifier method)

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    return
    for i in range(len(guesses)):
        prediction = guesses[i]
        truth = testLabels[i]
        if (prediction != truth):
            print "==================================="
            print "Mistake on example %d" % i
            print "Predicted %s; truth is %s" % (prediction, truth)
            print "Image: "
            print rawTestData[i]
            # break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
        Prints a Datum object that contains all pixels in the
        provided list of pixels.  This will serve as a helper function
        to the analysis function you write.

        Pixels should take the form
        [(2,2), (2, 3), ...]
        where each tuple represents a pixel.
        """
        image = samples.Datum(None,self.width,self.height)
        for pix in pixels:
            try:
            # This is so that new features that you could define which
            # which are not of the form of (x,y) will not break
            # this image printer...
                x,y = pix
                image.pixels[x][y] = 2
            except:
                print "new features:", pix
                continue
        print image

def default(str):
    return str + ' [Default: %default]'

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """


def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron', 'mira', 'minicontest'], default='mostFrequent')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces', 'pacman'], default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
    parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
    parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
    parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
    parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
    parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
    parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    parser.add_option('-g', '--agentToClone', help=default("Pacman agent to copy"), default=None, type="str")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print "Doing classification"
    print "--------------------"
    print "data:\t\t" + options.data
    print "classifier:\t\t" + options.classifier
    if not options.classifier == 'minicontest':
        print "using enhanced features?:\t" + str(options.features)
    else:
        print "using minicontest feature extractor"
    print "training set size:\t" + str(options.training)
    if(options.data=="digits"):
        printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorDigit
        else:
            featureFunction = basicFeatureExtractorDigit
        if (options.classifier == 'minicontest'):
            featureFunction = contestFeatureExtractorDigit
    elif(options.data=="faces"):
        printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorFace
        else:
            featureFunction = basicFeatureExtractorFace
    elif(options.data=="pacman"):
        printImage = None
        if (options.features):
            featureFunction = enhancedFeatureExtractorPacman
        else:
            featureFunction = basicFeatureExtractorPacman
    else:
        print "Unknown dataset", options.data
        print USAGE_STRING
        sys.exit(2)

    if(options.data=="digits"):
        legalLabels = range(10)
    else:
        legalLabels = ['Stop', 'West', 'East', 'North', 'South']

    if options.training <= 0:
        print "Training set size should be a positive integer (you provided: %d)" % options.training
        print USAGE_STRING
        sys.exit(2)

    if options.smoothing <= 0:
        print "Please provide a positive number for smoothing (you provided: %f)" % options.smoothing
        print USAGE_STRING
        sys.exit(2)

    if options.odds:
        if options.label1 not in legalLabels or options.label2 not in legalLabels:
            print "Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2)
            print USAGE_STRING
            sys.exit(2)

    if(options.classifier == "mostFrequent"):
        classifier = mostFrequent.MostFrequentClassifier(legalLabels)
    elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
        classifier.setSmoothing(options.smoothing)
        if (options.autotune):
            print "using automatic tuning for naivebayes"
            classifier.automaticTuning = True
        else:
            print "using smoothing parameter k=%f for naivebayes" %  options.smoothing
    elif(options.classifier == "perceptron"):
        if options.data != 'pacman':
            classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
        else:
            classifier = perceptron_pacman.PerceptronClassifierPacman(legalLabels,options.iterations)
    elif(options.classifier == "mira"):
        if options.data != 'pacman':
            classifier = mira.MiraClassifier(legalLabels, options.iterations)
        if (options.autotune):
            print "using automatic tuning for MIRA"
            classifier.automaticTuning = True
        else:
            print "using default C=0.001 for MIRA"
    elif(options.classifier == 'minicontest'):
        import minicontest
        classifier = minicontest.contestClassifier(legalLabels)
    else:
        print "Unknown classifier:", options.classifier
        print USAGE_STRING

        sys.exit(2)

    args['agentToClone'] = options.agentToClone

    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = printImage

    return args, options

# Dictionary containing full path to .pkl file that contains the agent's training, validation, and testing data.
MAP_AGENT_TO_PATH_OF_SAVED_GAMES = {
    'FoodAgent': ('pacmandata/food_training.pkl','pacmandata/food_validation.pkl','pacmandata/food_test.pkl' ),
    'StopAgent': ('pacmandata/stop_training.pkl','pacmandata/stop_validation.pkl','pacmandata/stop_test.pkl' ),
    'SuicideAgent': ('pacmandata/suicide_training.pkl','pacmandata/suicide_validation.pkl','pacmandata/suicide_test.pkl' ),
    'GoodReflexAgent': ('pacmandata/good_reflex_training.pkl','pacmandata/good_reflex_validation.pkl','pacmandata/good_reflex_test.pkl' ),
    'ContestAgent': ('pacmandata/contest_training.pkl','pacmandata/contest_validation.pkl', 'pacmandata/contest_test.pkl' )
}
# Main harness code



def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']

    # Load data
    numTraining = options.training
    numTest = options.test

    if(options.data=="pacman"):
        agentToClone = args.get('agentToClone', None)
        trainingData, validationData, testData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES.get(agentToClone, (None, None, None))
        trainingData = trainingData or args.get('trainingData', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][0]
        validationData = validationData or args.get('validationData', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][1]
        testData = testData or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][2]
        rawTrainingData, trainingLabels = samples.loadPacmanData(trainingData, numTraining)
        rawValidationData, validationLabels = samples.loadPacmanData(validationData, numTest)
        rawTestData, testLabels = samples.loadPacmanData(testData, numTest)
    else:
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
        rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
        rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)


    # Extract features
    print "Extracting features..."
    trainingData = map(featureFunction, rawTrainingData)
    validationData = map(featureFunction, rawValidationData)
    testData = map(featureFunction, rawTestData)

    # Conduct training and testing
    print "Training..."
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print "Validating..."
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels))
    print "Testing..."
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
    analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

    # do odds ratio computation if specified at command line
    if((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb")) ):
        label1, label2 = options.label1, options.label2
        features_odds = classifier.findHighOddsFeatures(label1,label2)
        if(options.classifier == "naiveBayes" or options.classifier == "nb"):
            string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
        else:
            string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)

        print string3
        printImage(features_odds)

    if((options.weights) & (options.classifier == "perceptron")):
        for l in classifier.legalLabels:
            features_weights = classifier.findHighWeightFeatures(l)
            print ("=== Features with high weight for label %d ==="%l)
            printImage(features_weights)

if __name__ == '__main__':
    # Read input
    args, options = readCommand( sys.argv[1:] )
    # Run classifier
    runClassifier(args, options)
