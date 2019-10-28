import cv2
import numpy as np


def detect_blob(src_img):
    # 斑点检测

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 20
    params.maxThreshold = 220
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100
    # params.maxArea = 800
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.01
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.75
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(src_img)
    img_with_points = cv2.drawKeypoints(src_img, keypoints, np.array([]), (255, 0, 0),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("keypoints", img_with_points)


def binaryzation(frame):
    lower = np.array([0, 100, 100])
    upper = np.array([120, 255, 255])

    out = cv2.inRange(frame, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.erode(out, kernel, iterations=1)
    # out = cv2.dilate(out, kernel, iterations=2)

    return frame, out


def findContours_img(original_img, out):
    contours, hierarchy = cv2.findContours(out, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    while True:
        if len(contours) == 0:
            pass
        else:
            break

    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    point = rect[0]
    x, y = rect[1]
    x0 = int(point[0] - x/2)
    x1 = int(point[0] + x/2)
    y0 = int(point[1] - y/2)
    y1 = int(point[1] + y/2)
    cut_img = original_img[y0:y1, x0:x1]

    mask_img = np.zeros(original_img.shape, np.uint8)

    # drawContour要求输进的列表
    c_max = []
    c_max.append(c)

    contour_img = cv2.drawContours(mask_img, c_max, -1, (255, 255, 255), cv2.FILLED)    #画出轮廓
    # draw_img2 = cv2.rectangle(original_img, (x0, y0), (x1, y1), (0, 0, 255), 2, cv2.LINE_8, 0)

    new_img = cv2.bitwise_and(original_img, mask_img)

    return cut_img, contour_img, new_img, mask_img


capture = cv2.VideoCapture(1)
while True:
    ref, frame = capture.read()
    frame = frame[:, 160:500]
    frame, out = binaryzation(frame)
    # cv2.imshow("bin", out)
    # cv2.waitKey(0)
    # 裁剪橘子
    cut_img, draw_img1, crop_img, mask_img = findContours_img(frame, out)
    cv2.imshow("crop_img", crop_img)
    cv2.imshow("mask_img", mask_img)

    if not len(cut_img) == 0:
        detect_blob(crop_img)

    # 等待30ms显示图像，若过程中按“Esc”退出
    c = cv2.waitKey(1) & 0xff
    if c == 27:
        capture.release()
        break

cv2.waitKey()
cv2.destroyAllWindows()
