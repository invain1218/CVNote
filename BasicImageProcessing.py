import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy

img = cv2.imread('image.jpg',cv2.IMREAD_COLOR) # BGR
print(img.shape)
# (1200, 1200, 3)  # 高度，宽度，通道数
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR to RGB
r,g,b = cv2.split(img_rgb) # 分离通道
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1) 
plt.imshow(img)
plt.title("Original (BGR shown as RGB)") 
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_rgb)
plt.title("Converted (BGR to RGB)")
plt.axis('off')
plt.tight_layout()
plt.show()


# 传统图像处理（PS）
# 灰度化
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)   
plt.figure(figsize=(5, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original (RGB)") 
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_gray,cmap='gray')
plt.title("Gray")
plt.axis('off')
plt.tight_layout()
plt.show()

img_16 = cv2.resize(img_rgb, (16, 16)) # resize to 16x16  
plt.figure(figsize=(5, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original (RGB)") 
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_16,cmap='gray')
plt.title("Resized (16x16)")
plt.axis('off')
plt.tight_layout()
plt.show()

img_crop = img_rgb[50:200, 50:200] # crop a region of interest
plt.figure(figsize=(5, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original (RGB)") 
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_crop,cmap='gray')
plt.title("Cropped (150x150)")
plt.axis('off')
plt.tight_layout()
plt.show()

# 图像增强
# 直方图均衡化
img_eq = cv2.equalizeHist(img_gray)
plt.figure(figsize=(5, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title("Original Gray")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_eq, cmap='gray')
plt.title("Equalized")
plt.axis('off')
plt.tight_layout()
plt.show()

# 低通滤波LPF：去噪
# 添加高斯噪声
import numpy as np

img_gau = img_gray.copy()
noise_gau = np.random.normal(0, 50, img_gau.shape)  
img_gau = img_gau.astype(np.float64) + noise_gau     # 先转 float 再加
img_gau = np.clip(img_gau, 0, 255).astype(np.uint8)   # 截断到 0-255

# 添加椒盐噪声 
img_sp = img_gray.copy()
rand = np.random.rand(*img_gray.shape)
img_sp[rand < 0.04] = 0      # 盐噪声（黑点）
img_sp[rand > 0.96] = 255    # 椒噪声（白点）

# 添加斑点噪声
noise_speckle = np.random.randn(*img_gray.shape)       # 标准正态
img_speckle = img_gray.astype(np.float64) + img_gray.astype(np.float64) * noise_speckle * 0.5
img_speckle = np.clip(img_speckle, 0, 255).astype(np.uint8)

# 添加周期噪声
rows, cols = img_gray.shape
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
pattern = 30 * np.sin(2 * np.pi * 20 * X / cols)       # 水平正弦条纹
img_periodic = img_gray.astype(np.float64) + pattern
img_periodic = np.clip(img_periodic, 0, 255).astype(np.uint8)

# 添加混合噪声
img_mixed = img_gray.astype(np.float64) + noise_gau
img_mixed = np.clip(img_mixed, 0, 255).astype(np.uint8)
rand_m = np.random.rand(*img_gray.shape)
img_mixed[rand_m < 0.04] = 0
img_mixed[rand_m > 0.96] = 255


# 均值滤波：g(i,j) = 1/(m·n) * ΣΣ f(i+k,j+l) 
# 用m×n邻域内所有像素的平均值替代（i,j）位置的像素。
# 适合去除高斯噪声，但会模糊边缘。OpenCV实现的均值滤波是基于卷积的。
img_mean_denoised = cv2.blur(img_periodic, (5, 5)) # kernel size = 5x5

# 高斯滤波：g(i,j) = 1/W * ΣΣ f(i+k,j+l) * exp(-(k²+l²)/(2σ²)), W = ΣΣexp(-(k^2+l^2)/(2σ^2)))  
# 用高斯函数加权邻域内像素值的平均值替代（i,j）位置的像素。
# 适合去除高斯噪声，保留边缘。OpenCV实现的高斯滤波是基于卷积的。
img_gau_denoised = cv2.GaussianBlur(img_gau, (5, 5), 0) # kernel size = 5x5, sigma = 0

# 中值滤波：g(i,j) = median{f(i+k,j+l)} 
# 用m×n邻域内像素值的中位数替代（i,j）位置的像素。
# 适合去除椒盐噪声，保留边缘。OpenCV实现的中值滤波是基于排序的。
img_median_denoised = cv2.medianBlur(img_sp, 5) # kernel size = 5x5

# 双边滤波：g(i,j) = 1/W * ΣΣ f(i+k,j+l) * exp(-(k²+l²)/(2σs²)) * exp(-(f(i+k,j+l)-f(i,j))²/(2σr²)) 
# 用空间高斯核和值域高斯核的乘积作为权重，对邻域内像素值进行加权求和，替代中心像素。权重同时考虑空间距离和像素值相似性——距离越近、值越接近，权重越大。
# 适合去除椒盐噪声，保留边缘。OpenCV实现的双边滤波是基于像素的。
img_bilateral = cv2.bilateralFilter(img_speckle, 9, 75, 75) # diameter = 9, sigmaColor = 75, sigmaSpace = 75

# 非局部均值滤波NLM：g(i,j) = 1/W * ΣΣ f(k,l) * exp(-||f(N(i,j))-f(N(k,l))||²/(2σ²)), W = ΣΣexp(-||f(N(i,j))-f(N(k,l))||²/(2σ²))
# 用图像中所有像素的加权平均值替代（i,j）
# 适合去除高斯噪声，保留图像细节。OpenCV实现的NLM是基于块的非局部均值滤波。
img_nlm = cv2.fastNlMeansDenoising(img_mixed, None, 10, 7, 21) # h = 10, templateWindowSize = 7, searchWindowSize = 21


plt.figure(figsize=(18, 8))
plt.subplot(2, 5, 1)
plt.imshow(img_gau, cmap='gray')
plt.title("Gaussian Noise")
plt.axis('off')

plt.subplot(2, 5, 2)
plt.imshow(img_sp, cmap='gray')
plt.title("Salt & Pepper Noise")
plt.axis('off')

plt.subplot(2, 5, 3)
plt.imshow(img_speckle, cmap='gray')
plt.title("Speckle Noise")
plt.axis('off')

plt.subplot(2, 5, 4)
plt.imshow(img_periodic, cmap='gray')
plt.title("Periodic Noise")
plt.axis('off')

plt.subplot(2, 5, 5)
plt.imshow(img_mixed, cmap='gray')
plt.title("Mixed Noise")
plt.axis('off')

plt.subplot(2, 5, 6)
plt.imshow(img_gau_denoised, cmap='gray')
plt.title("Gaussian Filter")
plt.axis('off')

plt.subplot(2, 5, 7)
plt.imshow(img_median_denoised, cmap='gray')
plt.title("Median Filter")
plt.axis('off')

plt.subplot(2, 5, 8)
plt.imshow(img_bilateral, cmap='gray')
plt.title("Bilateral Filter")
plt.axis('off')

plt.subplot(2, 5, 9)
plt.imshow(img_mean_denoised, cmap='gray')
plt.title("Mean Filter")
plt.axis('off')

plt.subplot(2, 5, 10)
plt.imshow(img_nlm, cmap='gray')
plt.title("NLM Filter")
plt.axis('off')

plt.tight_layout()
plt.show()


# 高通滤波HPF：边缘检测（提取边缘，一般来说边缘的像素值高于周围像素值）
# Sobel 算子
#        Gx = [[-1,0,1],[-2,0,2],[-1,0,1]] * f,  Gy = [[-1,-2,-1],[0,0,0],[1,2,1]] * f
#        G = sqrt(Gx² + Gy²)
# 一阶差分，分别计算水平和垂直方向梯度，对噪声有一定平滑作用
img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
img_sobel = np.sqrt(img_sobel_x**2 + img_sobel_y**2)
img_sobel = np.clip(img_sobel, 0, 255).astype(np.uint8)


# Laplacian 算子
#    ∇²f = ∂²f/∂x² + ∂²f/∂y² ≈ [[0,1,0],[1,-4,1],[0,1,0]] * f
# 二阶差分，对噪声敏感，常配合高斯滤波使用（LoG）
img_laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
img_laplacian = np.clip(np.abs(img_laplacian), 0, 255).astype(np.uint8)


# Prewitt 算子
# 公式：Gx = [[-1,0,1],[-1,0,1],[-1,0,1]] * f,  Gy = [[-1,-1,-1],[0,0,0],[1,1,1]] * f
# 与 Sobel 类似，但平滑权重相同（无中心加权），对噪声更敏感
kernel_prewitt_x = np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]], dtype=np.float32)
kernel_prewitt_y = np.array([[-1, -1, -1],
                              [ 0,  0,  0],
                              [ 1,  1,  1]], dtype=np.float32)
img_prewitt_x = cv2.filter2D(img_gray, cv2.CV_64F, kernel_prewitt_x)
img_prewitt_y = cv2.filter2D(img_gray, cv2.CV_64F, kernel_prewitt_y)
img_prewitt = np.sqrt(img_prewitt_x**2 + img_prewitt_y**2)
img_prewitt = np.clip(img_prewitt, 0, 255).astype(np.uint8)


# Scharr 算子
#    Gx = [[-3,0,3],[-10,0,10],[-3,0,3]] * f,  Gy = [[-3,-10,-3],[0,0,0],[3,10,3]] * f
# Sobel 的改进版，中心权重更大，对小边缘和细节更敏感
img_scharr_x = cv2.Scharr(img_gray, cv2.CV_64F, 1, 0)
img_scharr_y = cv2.Scharr(img_gray, cv2.CV_64F, 0, 1)
img_scharr = np.sqrt(img_scharr_x**2 + img_scharr_y**2)
img_scharr = np.clip(img_scharr, 0, 255).astype(np.uint8)


# Canny 算子
# 流程：高斯滤波 → Sobel 梯度 → 非极大值抑制 → 双阈值检测 → 边缘连接
# 输出为二值图（0 或 255），边缘细、连续、定位准
img_canny = cv2.Canny(img_gray, 100, 200)

plt.figure(figsize=(18, 8))

plt.subplot(2, 3, 1)
plt.imshow(img_gray, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_sobel, cmap='gray')
plt.title("Sobel")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(img_laplacian, cmap='gray')
plt.title("Laplacian")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img_prewitt, cmap='gray')
plt.title("Prewitt")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(img_scharr, cmap='gray')
plt.title("Scharr")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(img_canny, cmap='gray')
plt.title("Canny")
plt.axis('off')

plt.tight_layout()
plt.show()