import cv2
import numpy as np

def run_orb_experiment(nfeatures):
    # 读取图像
    img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print("❌ 图片读取失败！")
        return None

    # 1. ORB特征检测
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        print(f"❌ 未检测到描述子！(nfeatures={nfeatures})")
        return None

    # 2. 特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 3. RANSAC估计Homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if mask is None:
        print(f"❌ 无法估计Homography！(nfeatures={nfeatures})")
        return None

    # 统计数据
    kp1_count = len(kp1)
    kp2_count = len(kp2)
    match_count = len(matches)
    inlier_count = int(np.sum(mask))
    inlier_ratio = inlier_count / match_count
    success = "是" if H is not None else "否"

    return {
        "nfeatures": nfeatures,
        "模板图关键点": kp1_count,
        "场景图关键点": kp2_count,
        "匹配数量": match_count,
        "RANSAC内点数": inlier_count,
        "内点比例": round(inlier_ratio, 3),
        "是否成功定位": success
    }

# 测试三组参数
results = []
for n in [500, 1000, 2000]:
    res = run_orb_experiment(n)
    if res is not None:
        results.append(res)

# 打印结果表格
print("\n=== 任务6 参数对比实验结果 ===")
print(f"{'nfeatures':<12}{'模板图关键点':<14}{'场景图关键点':<14}{'匹配数量':<10}{'RANSAC内点数':<12}{'内点比例':<10}{'是否成功定位':<12}")
print("-" * 90)
for r in results:
    print(f"{r['nfeatures']:<12}{r['模板图关键点']:<14}{r['场景图关键点']:<14}{r['匹配数量']:<10}{r['RANSAC内点数']:<12}{r['内点比例']:<10}{r['是否成功定位']:<12}")