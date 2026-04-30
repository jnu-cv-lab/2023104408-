import cv2
import numpy as np
import os

# -------------------------- 1. 检查文件是否存在 --------------------------
print("=== 1. 检查文件 ===")
print("当前工作目录:", os.getcwd())
print("是否存在 box.png:", os.path.exists("box.png"))
print("是否存在 box_in_scene.png:", os.path.exists("box_in_scene.png"))

# -------------------------- 2. 读取图像 --------------------------
print("\n=== 2. 读取图像 ===")
img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

if img1 is None:
    print("❌ 错误：无法读取 box.png，请检查文件路径！")
if img2 is None:
    print("❌ 错误：无法读取 box_in_scene.png，请检查文件路径！")

if img1 is None or img2 is None:
    input("按回车退出...")
    exit()

print("✅ 两张图片读取成功！")

# -------------------------- 3. 创建 ORB 检测器 --------------------------
print("\n=== 3. 创建 ORB 检测器 ===")
orb = cv2.ORB_create(nfeatures=1000)
print("✅ ORB 检测器创建成功！")

# -------------------------- 4. 检测关键点和描述子 --------------------------
print("\n=== 4. 检测特征点 ===")
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
print("✅ 特征点检测完成！")

# -------------------------- 5. 可视化关键点 --------------------------
print("\n=== 5. 生成可视化图片 ===")
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)

cv2.imwrite('box_keypoints.png', img1_kp)
cv2.imwrite('box_in_scene_keypoints.png', img2_kp)
print("✅ 特征点图片已保存到桌面！")

# -------------------------- 6. 输出结果 --------------------------
print("\n=== 6. 实验结果 ===")
print("box.png 关键点数量：", len(kp1))
print("box_in_scene.png 关键点数量：", len(kp2))
print("描述子维度：", des1.shape[1] if des1 is not None else 0)

# 显示图片（可选）
cv2.imshow("box.png 特征点", img1_kp)
cv2.imshow("box_in_scene.png 特征点", img2_kp)
print("\n按任意键关闭图片窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()

input("\n✅ 程序运行完毕！按回车退出...")
