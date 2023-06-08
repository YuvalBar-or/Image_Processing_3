import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
import math
from typing import List


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 214329633

# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    # checking greyscale
    if len(im1.shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Calculate the derivatives in x and y directions
    vector = np.array([[1, 0, -1]])
    I_X = cv2.filter2D(im2, -1, vector, borderType=cv2.BORDER_REPLICATE)
    I_Y = cv2.filter2D(im2, -1, vector.T, borderType=cv2.BORDER_REPLICATE)
    I_T = im2 - im1

    # Initialize arrays for storing u, v and corresponding points
    u_v = []
    x_y = []

    # Iterate over the image with given step size
    for i in range(step_size, im1.shape[0], step_size):
        for j in range(step_size, im1.shape[1], step_size):
            # Create a small sample of I_X, I_Y, and I_T for the window around the current point
            sample_I_X = I_X[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1]
            sample_I_Y = I_Y[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1]
            sample_I_T = I_T[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1]

            # Flatten the sample arrays
            sample_I_X = sample_I_X.flatten()
            sample_I_Y = sample_I_Y.flatten()
            sample_I_T = sample_I_T.flatten()

            # Calculate the elements of A and B matrices
            sum_IX_squared = np.sum(sample_I_X ** 2)
            sum_IX_IY = np.sum(sample_I_X * sample_I_Y)
            sum_IY_squared = np.sum(sample_I_Y ** 2)

            sum_IX_IT = np.sum(sample_I_X * sample_I_T)
            sum_IY_IT = np.sum(sample_I_Y * sample_I_T)

            # Build A and B matrices
            A = np.array([[sum_IX_squared, sum_IX_IY], [sum_IX_IY, sum_IY_squared]])
            B = np.array([[-sum_IX_IT], [-sum_IY_IT]])

            # Calculate eigenvalues and eigenvectors of A
            eigen_val, eigen_vec = np.linalg.eig(A)
            eig_val1, eig_val2 = eigen_val

            # Check eigenvalue conditions
            if eig_val2 <= 1 or eig_val1 / eig_val2 >= 100:
                continue

            # Calculate u and v
            vector_u_v = np.dot(np.linalg.inv(A), B)
            u = -vector_u_v[0][0]
            v = -vector_u_v[1][0]

            x_y.append([j, i])
            u_v.append([u, v])

    return np.array(x_y), np.array(u_v)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    img1_pyr = gaussianPyr(img1, k)
    img2_pyr = gaussianPyr(img2, k)

    x_y_prev, u_v_prev = opticalFlow(img1_pyr[-1], img2_pyr[-1], stepSize, winSize)
    x_y_prev = list(x_y_prev)
    u_v_prev = list(u_v_prev)

    for i in range(1, k):
        x_y_i, uv_i = opticalFlow(img1_pyr[-1 - i], img2_pyr[-1 - i], stepSize, winSize)
        uv_i = list(uv_i)
        x_y_i = list(x_y_i)

        for g in range(len(x_y_i)):
            x_y_i[g] = list(x_y_i[g])

        for j in range(len(x_y_prev)):
            x_y_prev[j] = [element * 2 for element in x_y_prev[j]]
            u_v_prev[j] = [element * 2 for element in u_v_prev[j]]

        for j in range(len(x_y_i)):
            if x_y_i[j] in x_y_prev:
                index = x_y_prev.index(x_y_i[j])
                u_v_prev[index][0] += uv_i[j][0]
                u_v_prev[index][1] += uv_i[j][1]
            else:
                x_y_prev.append(x_y_i[j])
                u_v_prev.append(uv_i[j])

    arr3d = np.zeros(shape=(img1.shape[0], img1.shape[1], 2))

    for x in range(img1.shape[0]):
        for y in range(img1.shape[1]):
            if [y, x] not in x_y_prev:
                arr3d[x, y] = [0, 0]
            else:
                index = x_y_prev.index([y, x])
                arr3d[x, y] = u_v_prev[index]

    return arr3d

# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    u_v = opticalFlow(im1, im2, step_size=20, win_size=5)
    u = u_v[:, 0]
    v = u_v[:, 1]
    min_difference = float('inf')
    translation_matrix = np.array([[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]], dtype=float)
    for i in range(len(u)):
        t_ui = u[i]
        t_vi = v[i]

        # Create the matrix with current u, v
        translation_matrix_i = np.array([[1, 0, t_ui],
                                      [0, 1, t_vi],
                                      [0, 0, 1]], dtype=float)

        # Warp im1 using the current translation matrix
        img = cv2.warpPerspective(im1, translation_matrix_i, im1.shape[::-1])

        # Calculate the Mean Squared Error (MSE)
        mse = np.square(im2 - img).mean()

        # Check if the current MSE is smaller than the minimum difference and update accordingly
        if mse < min_difference:
            min_difference = mse
            translation_matrix = translation_matrix_i

    return translation_matrix


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    xy, uv = opticalFlow(im1, im2, step_size=20, win_size=5)
    xy_after_change = xy + uv
    angle_list = []
    xy_after_change = xy_after_change.astype(float)

    for i in range(len(xy)):
        first_angles_y, first_angles_x = xy[i][0] - (0, 0)[0], xy[i][1] - (0, 0)[1]
        second_angles_y, second_angles_x = xy_after_change[i][0] - (0, 0)[0], xy_after_change[i][1] - (0, 0)[1]
        arctan1 = math.atan2(first_angles_x, first_angles_y)
        arctan2 = math.atan2(second_angles_x, second_angles_y)
        if arctan1 < 0:
            arctan1 += math.pi
        if arctan2 < 0:
            arctan2 += math.pi
        if arctan1 <= arctan2:
            angle_list.append(arctan2 - arctan1)
        else:
            angle_list.append(math.pi / 3 + arctan2 - arctan1)

    angle_list = np.array(angle_list)
    theta = np.median(angle_list)

    mat_to_extract_xy_from = findTranslationCorr(im1, im2)
    t_x = mat_to_extract_xy_from[0][2]
    t_y = mat_to_extract_xy_from[1][2]

    translation_mat = np.float32([[np.cos(np.radians(theta)), -np.sin(np.radians(theta)), t_x],
                                      [np.sin(np.radians(theta)), np.cos(np.radians(theta)), t_y],
                                      [0, 0, 1]])

    return translation_mat


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    subtle_pading = np.max(im1.shape) // 2
    pading1 = np.fft.fft2(np.pad(im1, subtle_pading))
    pading2 = np.fft.fft2(np.pad(im2, subtle_pading))
    prod = pading1 * pading2.conj()
    result_full = np.fft.fftshift(np.fft.ifft2(prod))
    corr = result_full.real[1 + subtle_pading:-subtle_pading + 1, 1 + subtle_pading:-subtle_pading + 1]
    y1, x1 = np.unravel_index(np.argmax(corr), corr.shape)
    y2, x2 = np.array(im2.shape) // 2

    translation_mat = np.float32([[1, 0, x2 - x1 - 1], [0, 1, y2 - y1 - 1], [0, 0, 1]])
    return translation_mat


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    subtle_pading = np.max(im1.shape) // 2
    pading1 = np.fft.fft2(np.pad(im1, subtle_pading))
    pading2 = np.fft.fft2(np.pad(im2, subtle_pading))
    prod = pading1 * pading2.conj()
    result_full = np.fft.fftshift(np.fft.ifft2(prod))
    corr = result_full.real[1 + subtle_pading:-subtle_pading + 1, 1 + subtle_pading:-subtle_pading + 1]
    y1, x1 = np.unravel_index(np.argmax(corr), corr.shape)
    y2, x2 = np.array(im2.shape) // 2

    first_angles_y, first_angles_x = (x2, y2) - (0, 0)
    second_angles_y, second_angles_x = (x1, y1) - (0, 0)
    arctan1 = math.atan2(first_angles_x, first_angles_y)
    arctan2 = math.atan2(second_angles_x, second_angles_y)
    if arctan1 < 0:
        arctan1 += math.pi
    if arctan2 < 0:
        arctan2 += math.pi
    if arctan1 <= arctan2:
        theta = arctan2 - arctan1
    else:
        theta = math.pi / 3 + arctan2 - arctan1

    mat = np.float32([
        [np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0],
        [np.sin(np.radians(theta)), np.cos(np.radians(theta)), 0],
        [0, 0, 1]
    ])
    mat = np.linalg.inv(mat)
    rotate = cv2.warpPerspective(im2, mat, im2.shape[::-1])
    pading1 = np.fft.fft2(np.pad(im1, subtle_pading))
    pading2 = np.fft.fft2(np.pad(rotate, subtle_pading))
    prod = pading1 * pading2.conj()
    result_full = np.fft.fftshift(np.fft.ifft2(prod))
    corr = result_full.real[1 + subtle_pading:-subtle_pading + 1, 1 + subtle_pading:-subtle_pading + 1]
    y1, x1 = np.unravel_index(np.argmax(corr), corr.shape)
    y2, x2 = np.array(rotate.shape) // 2
    t_x = x2 - x1 - 1
    t_y = (y2 - y1 - 1) / 6

    translation_mat = np.float32([
        [np.cos(np.radians(theta)), -np.sin(np.radians(theta)), t_x],
        [np.sin(np.radians(theta)), np.cos(np.radians(theta)), t_y],
        [0, 0, 1]
    ])

    return translation_mat


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    # initiallize
    ret_img = np.zeros_like(im2)

    # Iterate over image 2
    for x in range(im2.shape[0]):
        for y in range(im2.shape[1]):
            pixel_3d = np.array([[x], [y], [1]])
            get_pixel_from_img1 = T @ pixel_3d
            img1_x = get_pixel_from_img1[0] / get_pixel_from_img1[2]
            img1_y = get_pixel_from_img1[1] / get_pixel_from_img1[2]

            # Check if pixels are ints or floats
            float_x = img1_x % 1
            float_y = img1_y % 1

            # If floats, transform them from im2
            if float_x != 0 or float_y != 0:
                floor_x = int(np.floor(img1_x))
                floor_y = int(np.floor(img1_y))
                ceil_x = int(np.ceil(img1_x))
                ceil_y = int(np.ceil(img1_y))
                ret_img[x, y] = ((1 - float_x) * (1 - float_y) * im2[floor_x, floor_y]) \
                                + (float_x * (1 - float_y) * im2[ceil_x, floor_y]) \
                                + (float_x * float_y * im2[ceil_x, ceil_y]) \
                                + ((1 - float_x) * float_y * im2[floor_x, ceil_y])
            # If they are ints, transform them as is
            else:
                img1_x = int(img1_x)
                img1_y = int(img1_y)
                ret_img[x, y] = im2[img1_x, img1_y]
    return ret_img

# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    res = []
    h = pow(2, levels) * (img.shape[0] // pow(2, levels))
    w = pow(2, levels) * (img.shape[1] // pow(2, levels))
    img = img[:h, :w]
    res.append(img)

    kernel_arr = ([5, 5])
    sigma = 0.3 * ((kernel_arr[0] - 1) * 0.5 - 1) + 0.8
    kernel = cv2.getGaussianKernel(kernel_arr[0], sigma)

    for i in range(1, levels):
        blur = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        img = blur[::2, ::2]
        res.append(img)

    return res


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pyr = []
    kernel_size = 5
    kernel_sigma = int(0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8)
    kernel = cv2.getGaussianKernel(kernel_size, kernel_sigma)
    kernel = (kernel * kernel.transpose()) * 4

    gaussian_pyr = gaussianPyr(img, levels)
    for i in range(levels - 1):
        pyr_img = gaussian_pyr[i + 1]
        extended_pic = np.zeros((pyr_img.shape[0] * 2, pyr_img.shape[1] * 2))
        extended_pic[::2, ::2] = pyr_img
        extended_level = cv2.filter2D(extended_pic, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        curr_level = gaussian_pyr[i] - extended_level
        pyr.append(curr_level)
    pyr.append(gaussian_pyr[-1])

    return pyr


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    kernel_size = 5
    lap_pyr_copy = lap_pyr.copy()
    kernel_sigma = int(0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8)
    kernel = cv2.getGaussianKernel(kernel_size, kernel_sigma)
    kernel = (kernel * kernel.transpose()) * 4

    cur_layer = lap_pyr[-1]
    for i in range(len(lap_pyr_copy) - 2, -1, -1):
        extended_pic = np.zeros((cur_layer.shape[0] * 2, cur_layer.shape[1] * 2))
        extended_pic[::2, ::2] = cur_layer
        cur_layer = cv2.filter2D(extended_pic, -1, kernel, borderType=cv2.BORDER_REPLICATE) + lap_pyr_copy[i]

    return cur_layer

def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """

    img_1 = cropToMultipleOf2(img_1, levels)
    img_2 = cropToMultipleOf2(img_2, levels)
    mask = cropToMultipleOf2(mask, levels)

    im_blend = np.zeros_like(img_1)

    if img_1.ndim == 3 or img_2.ndim == 3:
        for channel in range(img_1.shape[2]):
            part_im1 = img_1[:, :, channel]
            part_im2 = img_2[:, :, channel]
            part_mask = mask[:, :, channel]
            lp_reduce1 = laplaceianReduce(part_im1, levels)
            lp_reduce2 = laplaceianReduce(part_im2, levels)
            gauss_pyr = gaussianPyr(part_mask, levels)
            lp_ret = []
            for i in range(levels):
                curr_lap = gauss_pyr[i] * lp_reduce1[i] + (1 - gauss_pyr[i]) * lp_reduce2[i]
                lp_ret.append(curr_lap)
            im_blend[:, :, channel] = laplaceianExpand(lp_ret)

    else:
        lp_reduce1 = laplaceianReduce(img_1, levels)
        lp_reduce2 = laplaceianReduce(img_2, levels)
        gauss_pyr = gaussianPyr(mask, levels)
        lp_ret = []
        for i in range(levels):
            curr_lap = gauss_pyr[i] * lp_reduce1[i] + (1 - gauss_pyr[i]) * lp_reduce2[i]
            lp_ret.append(curr_lap)
        im_blend = laplaceianExpand(lp_ret)

    naive_blend = mask * img_1 + (1 - mask) * img_2

    return naive_blend, im_blend


def cropToMultipleOf2(img: np.ndarray, levels: int) -> np.ndarray:
    """
    Crop image to be a multiple of 2 based on pyramid levels
    :param img: Image
    :param levels: Pyramid depth
    :return: Cropped image
    """
    height, width = img.shape[:2]
    new_height = height - height % (2 ** levels)
    new_width = width - width % (2 ** levels)
    return img[:new_height, :new_width]
