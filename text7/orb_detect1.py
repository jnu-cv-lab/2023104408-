import cv2
import numpy as np

# 读取图片
img1 = cv2.imread('box.png', 0)
img2 = cv2.imread('box_in_scene.png', 0)

# 初始化 ORB
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 创建暴力匹配器 BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 匹配
matches = bf.match(des1, des2)

# 按距离从小到大排序
matches = sorted(matches, key=lambda x: x.distance)

# 输出总匹配数量
print("总匹配数量：", len(matches))

# 画出前 50 个匹配
img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

# 保存结果图
cv2.imwrite('orb_matches.png', img_match)

# 显示图片
cv2.imshow('匹配结果', img_match)
cv2.waitKey(0)
cv2.destroyAllWindows()