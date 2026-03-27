import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 任务1：读取测试图片 ----------------------
img = cv2.imread("test1.jpg")  # 把 test.jpg 换成你自己的图片路径
if img is None:
    print(" 图片读取失败，请检查路径是否正确！")
    exit()

# ---------------------- 任务2：输出图像基本信息 ----------------------
height, width, channels = img.shape
dtype = img.dtype
print(f" 图像尺寸：{width} × {height}")
print(f" 通道数：{channels}")
print(f" 数据类型：{dtype}")

# ---------------------- 任务3：显示原图 ----------------------
import matplotlib.pyplot as plt

# 用 Matplotlib 显示
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.show()

# ---------------------- 任务4：转换为灰度图并显示 ----------------------
# 必须先定义 gray_img
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(gray_img,cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

# ---------------------- 任务5：保存灰度图 ----------------------
cv2.imwrite("gray_test.jpg", gray_img)
print(" 灰度图已保存为 gray_test.jpg")

# ---------------------- 任务6：NumPy 简单操作 ----------------------
# 示例1：输出某个像素值（比如坐标 (100, 100)）
pixel_value = gray_img[100, 100]
print(f" 像素 (100,100) 的灰度值：{pixel_value}")

# 示例2：裁剪左上角 100×100 区域并保存
crop_img = img[0:100, 0:100]
cv2.imwrite("crop_test.jpg", crop_img)
print(" 裁剪区域已保存为 crop_test.jpg")
