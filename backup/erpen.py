import cv2
import numpy as np
from pathlib import Path
import sys
import os
import time

# Global paths
PREPROCESSED_DIR = "C:/erpen/ErpenAPI/preprocessed"
DETECTION_DIR = "C:/erpen/ErpenAPI/detection"  
#PREPROCESSED_DIR = "C:/Users/Lenovo/Desktop/ErpenAPIv4/preprocessed"
#DETECTION_DIR = "C:/Users/Lenovo/Desktop/ErpenAPIv4/detection"
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
    if processed_images[0] is None:  # Sadece ilk resmi kontrol edin
        print("not_found")
        return []
    start_time = time.time()

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(processed_images[0], None)  # Sadece ilk resmi kullanın
    if des1 is None:
        print("No features found in test image.")
        return []

    scores = {}
    max_matches = 0
    for preprocessed_image_path in Path(PREPROCESSED_DIR).glob('*.png'):
        preprocessed_image = cv2.imread(str(preprocessed_image_path), cv2.IMREAD_GRAYSCALE)
        kp2, des2 = orb.detectAndCompute(preprocessed_image, None)
        if des2 is None:
            continue

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        num_matches = len(matches)
        model_name = preprocessed_image_path.stem.split('-')[0]
        scores[model_name] = max(scores.get(model_name, 0), num_matches)
        #eklendi
        if num_matches > max_matches:
            max_matches = num_matches
            best_match_path = str(preprocessed_image_path)

    if best_match_path:
        highlight_differences(best_match_path, processed_images[0])
        #print(f"Farklılıklar işaretlendi: {best_match_path} ve {processed_images}")
    else:
        print("Uygun eşleşme bulunamadı.")
        #eklendi

    

    max_matches = max(max_matches, num_matches)

    if not scores:
        print("No matches found.")
        return []

    # Normalize scores to a percentage based on the maximum matches found
    normalized_scores = {name: (num / max_matches) * 100 for name, num in scores.items()}

    # Find the top match and its score
    top_match = max(normalized_scores, key=normalized_scores.get)
    top_match_score = normalized_scores[top_match]

    end_time = time.time()
    print(f"{end_time - start_time:.2f} saniye|")
    return [(top_match, top_match_score)]


from datetime import datetime



def highlight_differences(base_image_path, test_img):
    try:
        # Load the base image
        base_img = cv2.imread(base_image_path, cv2.IMREAD_GRAYSCALE)

        
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file_name = f"base_img_{current_time}.png"
        output_path = os.path.join(DETECTION_DIR, output_file_name)

        # Save the image with rectangles
        cv2.imwrite(output_path, base_img)
        if base_img is None or test_img is None:
            print("One or both images could not be loaded.")
            return

        # Resize images to the same size
        #base_img = cv2.resize(base_img, (256, 256))
        #test_img = cv2.resize(test_img, (256, 256))

        # Convert the test image to color for drawing red rectangles
        test_img_color = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file_name = f"test_img{current_time}.png"
        output_path = os.path.join(DETECTION_DIR, output_file_name)

        # Save the image with rectangles
        cv2.imwrite(output_path, test_img)

        # Compute the absolute difference between the two images
        diff = cv2.absdiff(base_img, test_img)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        white_pixels = np.sum(thresh == 255)
    
        # Count black pixels (where pixel value is 0)
        black_pixels = np.sum(thresh == 0)
        
        # Print the counts
        #print(f"White pixels: {white_pixels}, Black pixels: {black_pixels}")
        # Find contours of the differences
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Total white pixels in the entire test image
        total_white_pixels = np.sum(test_img == 255)

        # White pixels within the contours
        white_pixels_in_contours = 0
        roi_sayisi = 0
        max_roi = 0
        # Draw red rectangles around the differences
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi_num = w * h
            if roi_num > 100:
                cv2.rectangle(test_img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # Focus on the area inside the rectangle to count white pixels
                roi = thresh[y:y+h, x:x+w]
                white_pixels_in_contours += np.sum(roi == 255)
                roi_sayisi +=1
                roi_num = w * h
                if roi_num > max_roi:
                    max_roi=roi_num
                #print(white_pixels_in_contours,"---",roi_sayisi,"----",max_roi)

        ratio_of_white = (white_pixels_in_contours / total_white_pixels) * 100 if total_white_pixels > 0 else 0

        # Print the counts
        #print(f"White pixels in contours: {white_pixels_in_contours}, Total white pixels: {total_white_pixels}")
        #print(f"Ratio of white pixels within contours to total: {ratio_of_white:.2f}%")

        # Generate a unique file name using the current date and time
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file_name = f"detection_{current_time}.png"
        output_path = os.path.join(DETECTION_DIR, output_file_name)

        # Save the image with rectangles
        cv2.imwrite(output_path, test_img_color)
        #print(f"Marked image saved at {output_path}")
        if ratio_of_white >80:
            print("[]Hatalar Tespit Edilemedi! Lutfen Profilin Goruntu Kalibrasyonunu Saglayin|")
        elif ratio_of_white >10: #yüzde kaç hata oranı var
            print("Cok Kritik Hatalar Tespit Edildi|")
            print(f"{ratio_of_white:.2f}|")
        elif roi_sayisi > 10 or max_roi > 500:
            print("Cok Kritik Hatalar Tespit Edildi|")
            print(f"{ratio_of_white:.2f}|")
        elif roi_sayisi > 4 and roi_sayisi <10:
            print("Kritik Hatalar Tespit Edildi|")
            print(f"{ratio_of_white:.2f}|")
        elif roi_sayisi <= 4:
            print("Az Hata Tespit Edildi|")
            print(f"{ratio_of_white:.2f}|")
        elif roi_sayisi == 0 or max_roi < 50:
            print("Hatasiz İçerik|")
            print(f"{ratio_of_white:.2f}|")
        else:
            print("Hata tespitinde bir sorunla karsilasildi. Hatalar tespit edilemedi|")

    except Exception as e:
        print(f"An error occurred: {e}")








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
                print(f"%{score}")
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

