import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np

# Örnek resimleri yükleyelim (Burada yerel dosyalardan yüklemek yerine örnek olarak OpenCV'nin sağladığı verileri kullanacağım)
# Not: Gerçek uygulamada, 'resim1.jpg' ve 'resim2.jpg' gibi dosya yollarıyla resimlerinizi yükleyebilirsiniz.
img1 = cv2.imread(cv2.samples.findFile("stitching/a.jpg"))
img2 = cv2.imread(cv2.samples.findFile("stitching/b.jpg"))

# Resimleri griye çevir
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SSIM hesapla
score, diff = ssim(gray1, gray2, full=True)
print("SSIM:", score)

# SSIM farkını görselleştir
diff = (diff * 255).astype("uint8")
plt.imshow(diff, cmap='gray')
plt.title("Fark (SSIM)")
plt.show()
