import cv2
import numpy as np

# 1. 读取图像
img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)
img2_color = cv2.imread('box_in_scene.png')  # 用来画彩色框

if img1 is None or img2 is None or img2_color is None:
    print("❌ 图片读取失败！")
    exit()

# 2. ORB特征检测
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

if des1 is None or des2 is None:
    print("❌ 未检测到描述子！")
    exit()

# 3. 特征匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 4. RANSAC估计Homography
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 5. 目标定位：获取box.png的四个角点并投影
h, w = img1.shape
pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

# 使用cv2.perspectiveTransform()进行角点投影
dst = cv2.perspectiveTransform(pts, H)

# 使用cv2.polylines()在场景图中画出四边形边框
img_result = cv2.polylines(img2_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

# 6. 输出和保存结果
print("✅ 目标定位完成！")
cv2.imwrite('target_detection.png', img_result)
cv2.imshow("目标定位结果", img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()