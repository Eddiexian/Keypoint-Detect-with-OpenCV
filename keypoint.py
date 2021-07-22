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


def GFTT_kp_det(img):
    pts = cv2.goodFeaturesToTrack(
        img, maxCorners=300, qualityLevel=0.03, blockSize=3, minDistance=3)

    if pts is not None:
        kps = [cv2.KeyPoint(p[0][0], p[0][1], 3) for p in pts]
    else:
        kps = []

    return kps


def SIFT_kp_det(img):
    detector = cv2.SIFT_create()
    kps = detector.detect(img)

    return kps


def STAR_kp_det(img):
    detector = cv2.xfeatures2d.StarDetector_create()
    kps = detector.detect(img)

    return kps


def MSER_area_det(img):
    detector = cv2.MSER_create()
    kps = detector.detect(img)

    return kps


def BRISK_kp_det(img):
    detector = cv2.BRISK_create()
    kps = detector.detect(img)

    return kps


def ORB_kp_det(img):
    detector = cv2.ORB_create()
    kps = detector.detect(img)

    return kps


ORB_kp = ORB_kp_det(img_gray)
ORB_kp_img = cv2.drawKeypoints(img, ORB_kp, np.array(
    []), (0, 255, 0))
cv2.imshow("ORB_Keypoint", ORB_kp_img)

BRISK_kp = BRISK_kp_det(img_gray)
BRISK_kp_img = cv2.drawKeypoints(img, BRISK_kp, np.array(
    []), (0, 255, 0))
cv2.imshow("BRISK_Keypoint", BRISK_kp_img)


MSER_kp = MSER_area_det(img_gray)
MSER_kp_img = cv2.drawKeypoints(img, MSER_kp, np.array(
    []), (0, 255, 0))
cv2.imshow("MSER_Keypoint", MSER_kp_img)


STAR_kp = STAR_kp_det(img_gray)
STAR_kp_img = cv2.drawKeypoints(img, STAR_kp, np.array(
    []), (0, 255, 0))
cv2.imshow("STAR_Keypoint", STAR_kp_img)


SIFT_kp = SIFT_kp_det(img_gray)
SIFT_kp_img = cv2.drawKeypoints(img, SIFT_kp, np.array(
    []), (0, 255, 0))
cv2.imshow("SIFT_Keypoint", SIFT_kp_img)

GFTT_kp = GFTT_kp_det(img_gray)
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


# def SURF_kp_det(img):
#     detector = cv2.xfeatures2d_SURF.create(hessianThreshold=400)
#     kps = detector.detect(img)

#     return kps


# SURF_kp = SURF_kp_det(img_gray)
# SURF_kp_img = cv2.drawKeypoints(img, SURF_kp, np.array([]), (0, 255, 0))
# cv2.imshow("SURF_Keypoint", SURF_kp_img)
