import cv2
import numpy as np
from numpy.core.shape_base import block
from numpy.lib.type_check import imag

is_nms = True

img = cv2.imread("Lenna.jpg")
cv2.imshow("Source", img)
if img is None:
    print("Loading error!")
    exit()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def fast_kp_det(img, is_nms, threshold):

    fast_detector = cv2.FastFeatureDetector_create()

    if not is_nms:
        fast_detector.setNonmaxSuppression(0)

    # | center pixel - other pixel |
    fast_detector.setThreshold(threshold)

    kp = fast_detector.detect(img_gray)

    print("Fast Keypoint numbers : {}".format(len(kp)))
    return kp


def harris_kp_det(img, threshold):
    img = np.float32(img)
    H = cv2.cornerHarris(img, 2, 3, 0.04)
    kps = np.argwhere(H > threshold * H.max())
    kps = [cv2.KeyPoint(float(pt[1]), float(pt[0]), 3) for pt in kps]

    return kps


def GFTT(img):
    pts = cv2.goodFeaturesToTrack(
        img, maxCorners=300, qualityLevel=0.03, blockSize=3, minDistance=3)

    if pts is not None:
        kps = [cv2.KeyPoint(p[0][0], p[0][1], 3) for p in pts]
    else:
        kps = []

    return kps


GFTT_kp = GFTT(img_gray)
GFTT_kp_img = cv2.drawKeypoints(img, GFTT_kp, np.array([]), (0, 255, 0))
cv2.imshow("GFTT_Keypoint", GFTT_kp_img)

harris_kp = harris_kp_det(img_gray, 0.02)
harris_kp_img = cv2.drawKeypoints(img, harris_kp, np.array([]), (0, 255, 0))
cv2.imshow("Harris_Keypoint", harris_kp_img)

fast_kp = fast_kp_det(img_gray, is_nms, 20)
fast_kp_img = cv2.drawKeypoints(img, fast_kp, np.array(
    []), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
cv2.imshow("Fast_Keypoint", fast_kp_img)
cv2.waitKey(0)
