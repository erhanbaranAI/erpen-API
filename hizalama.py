import cv2
import numpy as np
from matplotlib import pyplot as plt

def align_images(im1, im2):
    # ORB detektörü kullanılıyor
    orb = cv2.ORB_create()

    # Anahtar noktaları ve tanımlayıcıları bul
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)

    # BF matcher oluştur ve eşleşmeleri hesapla
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Eşleşmeleri mesafeye göre sırala
    matches = sorted(matches, key=lambda x: x.distance)

    # En iyi eşleşmeleri kullanarak homografi matrisi hesapla
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # İlk resmi ikinci resme göre hizala
    h, w = im2.shape[:2]
    im1_aligned = cv2.warpPerspective(im1, M, (w, h))

    return im1_aligned

# Resimleri yükle
im1 = cv2.imread('r4_superkasa-2.png', 0)  # Gri tonlamalı olarak oku
im2 = cv2.imread('v1.png', 0)

# Resimleri hizala
im1_aligned = align_images(im1, im2)

# Hizalanmış resmi göster
plt.imshow(im1, cmap='gray')
plt.imshow(im1_aligned, cmap='gray')
plt.show()
