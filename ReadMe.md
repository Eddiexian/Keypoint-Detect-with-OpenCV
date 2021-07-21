# 特徵點檢測 OpenCV實作

## 1. Fast
* 以C為圓心尋找半徑R上的像素
* 若有n點大於或小於Cetner Pixel值
* 則該Center為Keypoint

```python=

def fast_kp_det(img, is_nms, threshold):

    fast_detector = cv2.FastFeatureDetector_create()

    if not is_nms:
        fast_detector.setNonmaxSuppression(0)

    # | center pixel - other pixel |
    fast_detector.setThreshold(threshold)

    kp = fast_detector.detect(img_gray)

    print("Fast Keypoint numbers : {}".format(len(kp)))
    return kp
    
```
![](https://i.imgur.com/epEvoHk.png)

## 2. Harris
* 根據window內的各pixel的X梯度Y梯度進行統計分析
* 將XY梯度變化映射到座標空間
* 利用空間中的的分布位置判定是否為角點
```python=
def harris_kp_det(img,threshold):
    img = np.float32(img)
    H = cv2.cornerHarris(img, 2, 3, 0.04)
    kps = np.argwhere(H > threshold * H.max())
    kps = [cv2.KeyPoint(float(pt[1]), float(pt[0]), 3) for pt in kps]

    return kps
```

![](https://i.imgur.com/2gCcVvj.png)

## 3. GFTT
* Harris的改良
* Harris同時考慮XY方向像梯度差是否大於閥值
* GFTT考慮XY中梯度差較小值是否大於閥值


```python=
def GFTT(img):
    pts = cv2.goodFeaturesToTrack(
        img, maxCorners=300, qualityLevel=0.03, blockSize=3, minDistance=3)

    if pts is not None:
        kps = [cv2.KeyPoint(p[0][0], p[0][1], 3) for p in pts]
    else:
        kps = []

    return kps
```
![](https://i.imgur.com/drd9LZw.png)
