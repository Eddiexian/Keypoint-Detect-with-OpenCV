import keypoint as kpdt
import cv2
import numpy as np


def SIFT_extract(img, kps):
    SIFT_extractor = cv2.SIFT_create()
    (kps, SIFT_dp) = SIFT_extractor.compute(img, kps, np.array([]))

    return SIFT_dp


def RootSIFT_extract(img, kps):

    SIFT_extractor = cv2.SIFT_create()
    (kps, SIFT_dp) = SIFT_extractor.compute(img, kps, np.array([]))

    if len(kps) > 0:
        # L1-正規化
        eps = 1e-7

        SIFT_dp /= (SIFT_dp.sum(axis=1, keepdims=True) + eps)
        # 取平方根
        SIFT_dp = np.sqrt(SIFT_dp)

    return SIFT_dp


imgA = cv2.imread("T3.jpg")
imgB = cv2.imread("T4.jpg")

if imgA is None:
    print("Loading Error")
    exit()

if imgB is None:
    print("Loading Error")
    exit()

imgA = cv2.resize(imgA, (512, 512), imgA)
imgB = cv2.resize(imgB, (512, 512), imgB)

cv2.imshow("SourceA", imgA)
cv2.imshow("SourceB", imgB)

img_grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
img_grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
# Get Keypoint
kpsA = kpdt.GFTT_kp_det(img_grayA)
kpsB = kpdt.GFTT_kp_det(img_grayB)
# Feature extract
dpA = RootSIFT_extract(img_grayA, kpsA)
dpB = RootSIFT_extract(img_grayB, kpsB)


matcher = cv2.DescriptorMatcher_create("BruteForce")
rawMatches = matcher.knnMatch(dpA, dpB, 2)

matches = []

for m in rawMatches:

    print("#1:{} , #2:{}".format(m[0].distance, m[1].distance))

    if len(m) == 2 and m[0].distance < m[1].distance * 0.8:

        matches.append((m[0].trainIdx, m[0].queryIdx))


(hA, wA) = imgA.shape[:2]
(hB, wB) = imgB.shape[:2]
vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
vis[0:hA, 0:wA] = imgA
vis[0:hB, wA:] = imgB

for (trainIdx, queryIdx) in matches:

    color = np.random.randint(0, high=255, size=(3,))

    ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))

    ptB = (int(kpsB[trainIdx].pt[0] + wA), int(kpsB[trainIdx].pt[1]))

    color = (int(color[0]), int(color[1]), int(color[2]))
    cv2.line(vis, ptA, ptB, color, 2)


cv2.imshow("vis", vis)
cv2.waitKey(0)
