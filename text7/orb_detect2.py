import cv2
import numpy as np

# -------------------------- 1. 读取图像 & ORB特征检测 --------------------------
img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("❌ 图片读取失败！")
    exit()

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

if des1 is None or des2 is None:
    print("❌ 未检测到描述子！")
    exit()

# -------------------------- 2. ORB特征匹配 --------------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

print("总匹配数量：", len(matches))

# -------------------------- 3. RANSAC剔除错误匹配 --------------------------
# 提取对应点坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 使用 cv2.findHomography() + RANSAC
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

print("\nHomography矩阵：")
print(H)

# 获取内点（RANSAC筛选后的匹配）
matches_mask = mask.ravel().tolist()
inlier_matches = [matches[i] for i in range(len(matches)) if matches_mask[i]]

# 输出结果
inlier_count = len(inlier_matches)
total_count = len(matches)
inlier_ratio = inlier_count / total_count

print("\nRANSAC内点数量：", inlier_count)
print("内点比例：{:.2f}".format(inlier_ratio))

# -------------------------- 4. 绘制RANSAC后的匹配结果 --------------------------
draw_params = dict(matchColor=(0, 255, 0),  # 匹配线颜色
                   singlePointColor=None,
                   matchesMask=matches_mask,  # 只画内点
                   flags=2)

img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

# 保存结果
cv2.imwrite('orb_ransac_matches.png', img_ransac)
print("\n✅ RANSAC匹配图已保存为 orb_ransac_matches.png！")

# 显示结果
cv2.imshow("RANSAC Matches", img_ransac)
cv2.waitKey(0)
cv2.destroyAllWindows()