import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def calculate_mse_psnr(original, recovered):
    """计算 MSE 和 PSNR"""
    mse = mean_squared_error(original.flatten(), recovered.flatten())
    psnr = 10 * np.log10(255**2 / mse)
    return mse, psnr

def downsample_image(img, method='direct', scale=0.5):
    """
    下采样图像
    method: 'direct' 直接下采样, 'gaussian' 先高斯平滑再下采样
    scale: 缩放比例
    """
    h, w = img.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    if method == 'direct':
        # 直接下采样（最近邻）
        downsampled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    elif method == 'gaussian':
        # 先高斯平滑再下采样
        blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
        downsampled = cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        downsampled = None
    
    return downsampled

def upsample_image(img, original_size, method='nearest'):
    """
    上采样恢复图像
    method: 'nearest', 'bilinear', 'bicubic'
    """
    h, w = original_size
    if method == 'nearest':
        upsampled = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    elif method == 'bilinear':
        upsampled = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    elif method == 'bicubic':
        upsampled = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        upsampled = None
    
    return upsampled

def compute_fft_spectrum(img):
    """计算并返回傅里叶变换的幅度谱（对数）"""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    log_magnitude = np.log(1 + magnitude_spectrum)
    # 归一化到0-255范围用于显示
    log_magnitude = cv2.normalize(log_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return log_magnitude.astype(np.uint8)

def compute_dct(img):
    """计算二维 DCT"""
    img_float = np.float32(img)
    dct = cv2.dct(img_float)
    return dct

def compute_low_frequency_energy_ratio(dct_coeffs, ratio=0.125):
    """
    计算低频区域能量占总能量的比例
    ratio: 低频区域占总区域的比例（取左上角正方形）
    """
    h, w = dct_coeffs.shape
    low_h, low_w = int(h * ratio), int(w * ratio)
    
    # 计算总能量（平方和）
    total_energy = np.sum(dct_coeffs**2)
    
    # 计算低频区域能量
    low_freq_energy = np.sum(dct_coeffs[:low_h, :low_w]**2)
    
    # 避免除以0
    if total_energy == 0:
        return 0
    
    return low_freq_energy / total_energy

def display_dct_coeffs(dct_coeffs, title):
    """显示 DCT 系数图（取对数）"""
    # 取绝对值并对数变换
    log_dct = np.log(1 + np.abs(dct_coeffs))
    # 归一化
    log_dct = cv2.normalize(log_dct, None, 0, 255, cv2.NORM_MINMAX)
    return log_dct.astype(np.uint8)

def main():
    # 1. 读入灰度图像
    # 请将 'your_image.jpg' 替换为你的图像路径
    img_path = 'test3.jpg'  # 修改为你的图像路径
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：无法读取图像，请检查路径！")
        return
    
    print(f"原始图像尺寸: {img.shape}")
    
    # 2. 下采样（缩小为1/2）
    scale = 0.5
    
    # 方法1：直接下采样
    img_down_direct = downsample_image(img, method='direct', scale=scale)
    
    # 方法2：高斯平滑再下采样
    img_down_gaussian = downsample_image(img, method='gaussian', scale=scale)
    
    print(f"下采样后尺寸: {img_down_direct.shape}")
    
    # 3. 图像恢复（使用不同的插值方法）
    # 对直接下采样的图像进行恢复
    restored_nearest = upsample_image(img_down_direct, img.shape, method='nearest')
    restored_bilinear = upsample_image(img_down_direct, img.shape, method='bilinear')
    restored_bicubic = upsample_image(img_down_direct, img.shape, method='bicubic')
    
    # 对高斯平滑后下采样的图像进行恢复
    restored_gaussian_nearest = upsample_image(img_down_gaussian, img.shape, method='nearest')
    restored_gaussian_bilinear = upsample_image(img_down_gaussian, img.shape, method='bilinear')
    restored_gaussian_bicubic = upsample_image(img_down_gaussian, img.shape, method='bicubic')
    
    # 4. 空间域比较 - 计算 MSE 和 PSNR
    print("\n" + "="*60)
    print("直接下采样后的恢复结果对比：")
    print("="*60)
    
    methods = ['最近邻', '双线性', '双三次']
    restored_images = [restored_nearest, restored_bilinear, restored_bicubic]
    
    for method, restored in zip(methods, restored_images):
        mse, psnr = calculate_mse_psnr(img, restored)
        print(f"{method}插值 - MSE: {mse:.4f}, PSNR: {psnr:.2f} dB")
    
    print("\n" + "="*60)
    print("高斯平滑下采样后的恢复结果对比：")
    print("="*60)
    
    restored_gaussian_images = [restored_gaussian_nearest, 
                                restored_gaussian_bilinear, 
                                restored_gaussian_bicubic]
    
    for method, restored in zip(methods, restored_gaussian_images):
        mse, psnr = calculate_mse_psnr(img, restored)
        print(f"{method}插值 - MSE: {mse:.4f}, PSNR: {psnr:.2f} dB")
    
    # 5. 显示图像
    plt.figure(figsize=(15, 12))
    
    # 显示原图和下采样图
    plt.subplot(3, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(img_down_direct, cmap='gray')
    plt.title('直接下采样')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(img_down_gaussian, cmap='gray')
    plt.title('高斯平滑后下采样')
    plt.axis('off')
    
    # 显示恢复结果
    plt.subplot(3, 4, 4)
    plt.imshow(restored_nearest, cmap='gray')
    plt.title('最近邻恢复（直接下采样）')
    plt.axis('off')
    
    plt.subplot(3, 4, 5)
    plt.imshow(restored_bilinear, cmap='gray')
    plt.title('双线性恢复（直接下采样）')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(restored_bicubic, cmap='gray')
    plt.title('双三次恢复（直接下采样）')
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.imshow(restored_gaussian_nearest, cmap='gray')
    plt.title('最近邻恢复（高斯下采样）')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(restored_gaussian_bilinear, cmap='gray')
    plt.title('双线性恢复（高斯下采样）')
    plt.axis('off')
    
    plt.subplot(3, 4, 9)
    plt.imshow(restored_gaussian_bicubic, cmap='gray')
    plt.title('双三次恢复（高斯下采样）')
    plt.axis('off')
    
    # 6. 傅里叶变换分析
    # 计算频谱
    fft_original = compute_fft_spectrum(img)
    fft_downsampled = compute_fft_spectrum(img_down_direct)
    fft_restored = compute_fft_spectrum(restored_bilinear)
    
    plt.subplot(3, 4, 10)
    plt.imshow(fft_original, cmap='gray')
    plt.title('原图频谱')
    plt.axis('off')
    
    plt.subplot(3, 4, 11)
    plt.imshow(fft_downsampled, cmap='gray')
    plt.title('下采样图频谱')
    plt.axis('off')
    
    plt.subplot(3, 4, 12)
    plt.imshow(fft_restored, cmap='gray')
    plt.title('双线性恢复图频谱')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 7. DCT 分析
    print("\n" + "="*60)
    print("DCT 低频能量比例分析（低频区域占比 1/8）：")
    print("="*60)
    
    # 计算 DCT
    dct_original = compute_dct(img)
    dct_restored_nearest = compute_dct(restored_nearest)
    dct_restored_bilinear = compute_dct(restored_bilinear)
    dct_restored_bicubic = compute_dct(restored_bicubic)
    
    # 计算低频能量比例
    ratio_original = compute_low_frequency_energy_ratio(dct_original, ratio=0.125)
    ratio_nearest = compute_low_frequency_energy_ratio(dct_restored_nearest, ratio=0.125)
    ratio_bilinear = compute_low_frequency_energy_ratio(dct_restored_bilinear, ratio=0.125)
    ratio_bicubic = compute_low_frequency_energy_ratio(dct_restored_bicubic, ratio=0.125)
    
    print(f"原始图像低频能量占比: {ratio_original*100:.2f}%")
    print(f"最近邻恢复图像低频能量占比: {ratio_nearest*100:.2f}%")
    print(f"双线性恢复图像低频能量占比: {ratio_bilinear*100:.2f}%")
    print(f"双三次恢复图像低频能量占比: {ratio_bicubic*100:.2f}%")
    
    # 显示 DCT 系数图
    dct_display_original = display_dct_coeffs(dct_original, '原图DCT')
    dct_display_nearest = display_dct_coeffs(dct_restored_nearest, '最近邻恢复DCT')
    dct_display_bilinear = display_dct_coeffs(dct_restored_bilinear, '双线性恢复DCT')
    dct_display_bicubic = display_dct_coeffs(dct_restored_bicubic, '双三次恢复DCT')
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(dct_display_original, cmap='gray')
    plt.title('原图 DCT 系数（对数）')
    plt.colorbar()
    
    plt.subplot(2, 2, 2)
    plt.imshow(dct_display_nearest, cmap='gray')
    plt.title('最近邻恢复 DCT 系数（对数）')
    plt.colorbar()
    
    plt.subplot(2, 2, 3)
    plt.imshow(dct_display_bilinear, cmap='gray')
    plt.title('双线性恢复 DCT 系数（对数）')
    plt.colorbar()
    
    plt.subplot(2, 2, 4)
    plt.imshow(dct_display_bicubic, cmap='gray')
    plt.title('双三次恢复 DCT 系数（对数）')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    # 8. 分析总结
    print("\n" + "="*60)
    print("分析总结：")
    print("="*60)
    print("""
    1. 傅里叶频谱分析：
       - 原图包含丰富的高频和低频信息
       - 下采样后高频成分减少，可能出现混叠现象
       - 恢复后高频成分有所恢复，但会引入插值产生的伪影
       
    2. DCT 能量分布分析：
       - 原始图像能量主要集中在低频区域
       - 最近邻插值会产生块状效应，DCT 高频系数较大
       - 双线性和双三次插值更平滑，能量更集中在低频
       - 双三次插值效果最好，PSNR 最高，能量最集中
       
    3. 不同插值方法比较：
       - 最近邻：速度快，但质量差，有锯齿
       - 双线性：质量较好，平滑，计算适中
       - 双三次：质量最好，但计算量最大
       
    4. 预滤波影响：
       - 先高斯平滑再下采样可以减少混叠
       - 但会损失更多高频细节
       - 恢复时 PSNR 可能略低，但视觉效果更自然
    """)

if __name__ == "__main__":
    main()