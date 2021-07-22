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

## 4. SIFT(檢測部分)
* 進行不同參數高斯模糊，獲得不同模糊度的圖片
* 進行直接的降採樣，獲得多分辨的圖片
* 將高斯模糊後的圖片進行相減操作(DOG)
* 對差分金字塔進行上下金字塔的比較，獲得極值點

```python=
def SIFT_kp_det(img):
    detector = cv2.SIFT_create()
    kps = detector.detect(img)

    return kps

```
![](https://i.imgur.com/aKDcRb1.png)

## 5.SURF
* 改進SIFT
* Hessian矩陣
* 新版Opencv已經移除(專利問題?)

## 6.STAR(CenSuRE)
* 原理較為複雜
* 使用不同角度的中心環繞濾波近似LoG

```python=
def STAR_kp_det(img):
    detector = cv2.xfeatures2d.StarDetector_create()
    kps = detector.detect(img)

    return kps
```
![](https://i.imgur.com/QNYLmef.png)

## 7.MSER
* 類似分水嶺算法
* 逐漸提高閥值(0~255)
* 高於閥值則塗黑，並在固定的閥值變化區間內取面積變化最小區域為最大穩定極值區域

```python=
def MSER_area_det(img):
    detector = cv2.MSER_create()
    kps = detector.detect(img)

    return kps
    
```
![](https://i.imgur.com/8bFqbx6.png)

## 8.BRISK
* FAST的強化版
* 創建影像金字塔，並針對各尺度進行FAST detect

```python=
def BRISK_kp_det(img):
    detector = cv2.BRISK_create()
    kps = detector.detect(img)

    return kp
```

![](https://i.imgur.com/aW4gdej.png)

## 9.ORB
* FAST強化版
* BRISK加上Harris篩選
```python=
def ORB_kp_det(img):
    detector = cv2.ORB_create()
    kps = detector.detect(img)

    return kps
```
![](https://i.imgur.com/7FZz7KR.png)

