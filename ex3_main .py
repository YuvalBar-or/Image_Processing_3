from ex3_utils import *
import time


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):
    print("LK Demo")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    pts, uv = opticalFlow(img_1.astype(float), img_2.astype(float), step_size=20, win_size=5)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print(np.median(uv,0))
    print(np.mean(uv,0))

    displayOpticalFlow(img_2, pts, uv)

def array3D(mat: np.ndarray) -> (np.ndarray, np.ndarray):
    x_y_return = []
    u_x_return = []
    for x in range(mat.shape[0]):
        for y in range(mat.shape[1]):
            if mat[x, y][0] != 0 or mat[x, y][1] != 0:
                x_y_return.append([y, x])
                u_x_return.append(mat[x, y])
    return np.array(x_y_return), np.array(u_x_return)

def hierarchicalkDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Hierarchical LK Demo")
    try:
        img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        img_1 = cv2.resize(img_1, (0, 0), fx=0.5, fy=0.5)
    except cv2.error as e:
        print("Error loading the image:", str(e))
        return

    t = np.array([[1, 0, -0.2],
                  [0, 1, -0.1],
                  [0, 0, 1]], dtype=float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    arr3d = opticalFlowPyrLK(img_1.astype(float), img_2.astype(float), stepSize=20, winSize=5, k=5)
    pts, uv = array3D(arr3d)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print(np.median(uv, 0))
    print(np.mean(uv, 0))

    displayOpticalFlow(img_2, pts, uv)


def compareLK(img_path):
    """
    ADD TEST
    Compare the two results from both functions.
    :param img_path: Image input
    :return:
    """
    print("Compare LK & Hierarchical LK")
    im = cv2.imread(img_path)
    img_1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=0.5, fy=0.5)
    t = np.array([[1, 0, -0.2],
                  [0, 1, -0.1],
                  [0, 0, 1]], dtype=float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    pts1, uv1 = opticalFlow(img_1.astype(float), img_2.astype(float), step_size=20, win_size=5)
    arr3d = opticalFlowPyrLK(img_1.astype(float), img_2.astype(float), stepSize=20, winSize=5, k=5)
    pts2, uv2 = array3D(arr3d)
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img_2, cmap='gray')
    ax[0].quiver(pts1[:, 0], pts1[:, 1], uv1[:, 0], uv1[:, 1], color='r')
    ax[0].set_title('lkDemo image')
    ax[1].imshow(img_2, cmap='gray')
    ax[1].quiver(pts2[:, 0], pts2[:, 1], uv2[:, 0], uv2[:, 1], color='r')
    ax[1].set_title('hierarchicalkDemo image')
    plt.show()
    cv2.waitKey(0)


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')

    plt.show()


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def imageWarpingDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Image Warping Demo")
    im = cv2.imread(img_path)
    img_path = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img_path = cv2.resize(img_path, (0, 0), fx=0.5, fy=0.5)
    t = np.array([[0.9, 0, 0],
                  [0, 0.9, 0],
                  [0, 0, 1]], dtype=float)
    img_2 = cv2.warpPerspective(img_path, t, img_path.shape[::-1])
    my_function = warpImages(img_path, img_2, t)
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(img_path, cmap='gray')
    ax[0].set_title('Original Photo')
    ax[1].imshow(img_2, cmap='gray')
    ax[1].set_title('Warped Image')
    ax[2].imshow(my_function, cmap='gray')
    ax[2].set_title('My Inverse Warp')
    plt.show()
    cv2.waitKey(0)


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):
    print("Gaussian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)

    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    canvas = np.zeros((canv_h, canv_w, 3))

    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]

    plt.imshow(canvas)
    plt.show()


def pyrLaplacianDemo(img_path):
    print("Laplacian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()


def blendDemo():
    im1 = cv2.cvtColor(cv2.imread('input/sunset .jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('input/cat .jpg'), cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread('input/mask_cat.jpg'), cv2.COLOR_BGR2RGB) / 255

    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)

    plt.show()

    cv2.imwrite('input/sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    print("ID:", myID())

    img_path = 'input/boxMan (1).jpg'
    #lkDemo(img_path)
    #hierarchicalkDemo(img_path)
    #compareLK(img_path)

    #imageWarpingDemo(img_path)

    #pyrGaussianDemo('input/pyr_bit.jpg')
    #pyrLaplacianDemo('input/pyr_bit.jpg')
    #blendDemo()


if __name__ == '__main__':
    main()
