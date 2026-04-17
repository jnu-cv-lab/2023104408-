import cv2
import numpy as np

def generate_test_image():
    # 创建 600x600 白色背景画布
    img = np.ones((600, 600, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 255), 3)
    cv2.circle(img, (450, 450), 80, (0, 255, 0), 3)
    cv2.line(img, (50, 150), (550, 150), (255, 0, 0), 2)
    cv2.line(img, (50, 200), (550, 200), (255, 200), 2)
    cv2.line(img, (250, 50), (250, 550), (0, 0, 0), 2)
    cv2.line(img, (300, 50), (300, 550), (0, 0, 0), 2)

    cv2.imwrite("test_image.png", img)
    print("✅ 测试图已保存为: test_image.png")
    return img

def similarity_transform(img):
    rows, cols = img.shape[:2]
    center = (cols // 2, rows // 2)
    angle = 30    
    scale = 0.8   
    M = cv2.getRotationMatrix2D(center, angle, scale)
    similar_img = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))
    cv2.imwrite("similar_transform.png", similar_img)
    print("✅ 相似变换结果已保存为: similar_transform.png")
    return similar_img

def affine_transform(img):
    rows, cols = img.shape[:2]
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [220, 60], [80, 260]])
    M = cv2.getAffineTransform(pts1, pts2)
    affine_img = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))
    cv2.imwrite("affine_transform.png", affine_img)
    print("✅ 仿射变换结果已保存为: affine_transform.png")
    return affine_img

def perspective_transform(img):
    rows, cols = img.shape[:2]
    pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    pts2 = np.float32([[100, 50], [cols-100, 100], [50, rows-50], [cols-50, rows-100]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    perspective_img = cv2.warpPerspective(img, M, (cols, rows), borderValue=(255, 255, 255))
    cv2.imwrite("perspective_transform.png", perspective_img)
    print("✅ 透视变换结果已保存为: perspective_transform.png")
    return perspective_img

def perspective_correction():
    img = cv2.imread("test5.jpg")
    if img is None:
        print("❌ 错误：找不到 test5.jpg，请确保文件和代码在同一目录！")
        return None

    rows, cols = img.shape[:2]
    pts_src = np.float32([
        [60, 220],    
        [729, 198],   
        [399, 1213],    
        [1279, 912]    
    ])
    target_w, target_h = 500, 707
    pts_dst = np.float32([
        [0, 0],
        [target_w, 0],
        [0, target_h],
        [target_w, target_h]
    ])

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    corrected_img = cv2.warpPerspective(img, M, (target_w, target_h), borderValue=(255,255,255))

    cv2.imwrite("corrected_test5.png", corrected_img)
    print("✅ test5.jpg 校正结果已保存为: corrected_test5.png")
    return corrected_img

if __name__ == "__main__":
    test_img = generate_test_image()
    cv2.imshow("Original Test Image", test_img)

    similar_img = similarity_transform(test_img)
    affine_img = affine_transform(test_img)
    perspective_img = perspective_transform(test_img)

    corrected_test5 = perspective_correction()

    cv2.imshow("Similar Transform", similar_img)
    cv2.imshow("Affine Transform", affine_img)
    cv2.imshow("Perspective Transform", perspective_img)
    if corrected_test5 is not None:
        cv2.imshow("Corrected test5.jpg", corrected_test5)

    print("\n所有图像已保存到当前目录：")
    print("test_image.png          - 原始测试图")
    print("similar_transform.png   - 相似变换结果")
    print("affine_transform.png    - 仿射变换结果")
    print("perspective_transform.png - 透视变换结果")
    print("corrected_test5.png     - 你的A4纸校正结果")

    cv2.waitKey(0)
    cv2.destroyAllWindows()