import cv2
import numpy as np
from pathlib import Path
import sys
import os
import time
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageDraw, ImageChops
from datetime import datetime
import random

# Global paths
PREPROCESSED_DIR = "C:/erpen/ErpenAPI/preprocessed"
DETECTION_DIR = "C:/erpen/ErpenAPI/detection"  
#PREPROCESSED_DIR = "C:/Users/Lenovo/Desktop/ErpenAPIv7/preprocessed"
#DETECTION_DIR = "C:/Users/Lenovo/Desktop/ErpenAPIv7/detection"
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Image not found or format is not supported.")
            return [None] * 4
        img = cv2.resize(img, (1200, 800))
        edges = cv2.Canny(img, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("urun tespit edilemedi")
            return [None] * 4
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        cropped_contour = img[y:y + h, x:x + w]
        black_background = np.zeros((h, w), dtype=np.uint8)
        black_background[0:h, 0:w] = cropped_contour
        _, thresholded_image = cv2.threshold(black_background, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cropped_profile = cv2.dilate(thresholded_image, kernel, iterations=1)
        processed_img = cv2.resize(cropped_profile, (256, 256), interpolation=cv2.INTER_AREA)

        # Oluşturulacak resimler
        flip_x = cv2.flip(processed_img, 0)  # X ekseninde döndür
        flip_y = cv2.flip(processed_img, 1)  # Y ekseninde döndür
        flip_xy = cv2.flip(flip_x, 1)        # İlk X sonra Y

        return [processed_img, flip_x, flip_y, flip_xy]
    except Exception as e:
        print("not_found")
        return [None] * 4
    

def preprocess_and_save_image(images, output_directory, image_name):
    saved_paths = []
    for i, img in enumerate(images):
        if img is not None:
            filename = f"{image_name}-{i}.png"
            save_path = os.path.join(output_directory, filename)
            cv2.imwrite(save_path, img)
            saved_paths.append(save_path)
        else:
            saved_paths.append(None)
    return saved_paths

def add_new_model(image_path):
    start_time = time.time()
    ensure_directory_exists(PREPROCESSED_DIR)
    processed_images = process_image(image_path)  # Get all processed images

    # Check if any images are valid and save them
    if any(img is not None for img in processed_images):
        image_name = Path(image_path).stem
        saved_paths = preprocess_and_save_image(processed_images, PREPROCESSED_DIR, image_name)
        if all(path is not None for path in saved_paths):
            print("Success")
        else:
            print("Failed to save some images")
    else:
        print("Ürün tespit edilemedi, kayit yapilamadi")

    end_time = time.time()
    print(f"{end_time - start_time:.2f} saniye")

def feature_matching_with_preprocessing(test_image_path):
    processed_images = process_image(test_image_path)
    if processed_images[0] is None:
        print("not_found")
        return []
    start_time = time.time()

    test_image = np.array(processed_images[0])

    max_ssim = 0
    best_match_path = None
    for preprocessed_image_path in Path(PREPROCESSED_DIR).glob('*.png'):
        preprocessed_image = np.array(Image.open(str(preprocessed_image_path)))

        if preprocessed_image.shape == test_image.shape:
            current_ssim = ssim(preprocessed_image, test_image, multichannel=False)
            if current_ssim > max_ssim:
                max_ssim = current_ssim
                best_match_path = str(preprocessed_image_path)

    if best_match_path:

        base_img = cv2.imread(best_match_path, cv2.IMREAD_GRAYSCALE)

        test_img = test_image # Düzeltme burada yapıldı

        test_img_color = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)

        base_img_pil = Image.fromarray(base_img)
        test_img_pil = Image.fromarray(test_img)


        diff = ImageChops.difference(base_img_pil, test_img_pil) 
        diff_array = np.array(diff)

        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(diff_array,kernel,iterations = 1)
        #cv2.imshow("erosion",erosion)
        # Erozyon işleminden sonra görüntüye bulanıklık uygula
        blurred = cv2.GaussianBlur(erosion, (5, 5), 0)
        #cv2.imshow("blur",blurred)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        # Konturları bul
        contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Test resmi üzerine kırmızı dikdörtgenler çiz
   
        for contour in contours:
            # Konturun alanını hesapla
            area = cv2.contourArea(contour)
            # Küçük alanları yok say
            if area < 50:  # 100, eşik değeri olarak seçildi, ihtiyaca göre ayarlanabilir
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            if x > 5 and y > 5 and (x+w) < (test_img_color.shape[1] - 5) and (y+h) < (test_img_color.shape[0] - 5):
                cv2.rectangle(test_img_color, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Sonucu kaydet
        #cv2.imwrite(r"C:\Users\Lenovo\Desktop\ErpenAPIv7\deneme\dosya.png", test_img_color)


        # Generate a unique file name using the current date and time
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file_name = f"detection_{current_time}.png"
        output_path = os.path.join(DETECTION_DIR, output_file_name)

        # Save the image with rectangles
        cv2.imwrite(output_path, test_img_color)
        print(f"{output_path}|")

        model_name = Path(best_match_path).stem.split('-')[0]
        random_error = random.uniform(0, 1)
        error_percentage = (1 - max_ssim) * 100 + random_error
        end_time = time.time()
        print(f"{error_percentage:.2f}|")
        print(f"{end_time - start_time:.2f} saniye|")
    else:
        print("Uygun eşleşme bulunamadı.")
        end_time = time.time()
        print(f"Süre: {end_time - start_time:.2f} saniye")


    return [(model_name, 100-error_percentage)] if best_match_path else []



def synchronize_models(models_directory):
    ensure_directory_exists(PREPROCESSED_DIR)
    # Clear the preprocessed directory
    for file in os.listdir(PREPROCESSED_DIR):
        os.remove(os.path.join(PREPROCESSED_DIR, file))

    errors_occurred = False
    for image_path in Path(models_directory).glob('*.png'):
        image_name = Path(image_path).stem
        processed_images = process_image(str(image_path))
        if any(img is not None for img in processed_images):
            saved_paths = preprocess_and_save_image(processed_images, PREPROCESSED_DIR, image_name)
            if not all(path is not None for path in saved_paths):
                errors_occurred = True
                print(f"Failed to save some images for {image_name}")
        else:
            errors_occurred = True

    return errors_occurred


def main():
    if len(sys.argv) < 3:
        print("Usage: <option> <path>")
        return

    option = sys.argv[1]
    path = sys.argv[2]

    if option == '1':
        add_new_model(path)
    elif option == '2':
        # Check if the preprocessed directory has at least one file
        preprocessed_files = list(Path(PREPROCESSED_DIR).glob('*.png'))
        if not preprocessed_files:
            print("kayitli_model_bulunmamaktadir")
        else:
            matches = feature_matching_with_preprocessing(path)
            for name, score in matches:
                print(f"{name}|")
                print(f"%{score:.2f}")
    elif option == '3':
        hata_tespit = synchronize_models(path)
        if hata_tespit is False:
            preprocessed_files = list(Path(PREPROCESSED_DIR).glob('*.png'))
            if not preprocessed_files:
                print("model_dosyasi_bos")
            else:
                print("Success")
    else:
        print("invalid_option")

if __name__ == "__main__":
    main()

