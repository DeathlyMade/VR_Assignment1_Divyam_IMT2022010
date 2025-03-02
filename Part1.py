import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply different types of blurring
    blurred = cv2.GaussianBlur(gray, (19, 19), 0)
    # equalized = cv2.medianBlur(gray, 5)
    # blurred = cv2.bilateralFilter(gray, 9, 25, 5)
    
    # min_val = np.min(gray)
    # max_val = np.max(gray)


    # stretched_image = (gray - min_val) / (max_val - min_val) * 255
    # stretched_image = np.uint8(stretched_image)

    # Image histogram equalization
    equalized = cv2.equalizeHist(blurred)
    
    # Apply thresholding
    #_, thresholded_img_equalized = cv2.threshold(equalized, 119, 255, cv2.THRESH_BINARY_INV)
    _, thresholded_img = cv2.threshold(blurred, 125, 255, cv2.THRESH_BINARY_INV)
    
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_img, connectivity=4)

    # min_size = 50  # Adjust this value based on the noise level
    # for i in range(0, num_labels):  # Skip background (0)
    #     if stats[i, cv2.CC_STAT_AREA] < min_size:
    #         thresholded_img[labels == i] = 255
    # Apply edge detection
    edges = cv2.Canny(thresholded_img, 50, 150)
    
    # kernel = np.ones((1, 1), np.uint8)
    # dilated = cv2.dilate(edges, kernel, iterations=2)

    # Apply Morphological Closing to close small gaps
    # closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of coins in the image: {len(contours)}")
    
    # Draw contours
    outlined = image.copy()
    for cnt in contours:
        cv2.drawContours(outlined, [cnt], 0, (0, 255, 0), thickness=2)
    
    filled = image.copy()
    for cnt in contours:
        cv2.drawContours(filled, [cnt], 0, (0, 255, 0), thickness=cv2.FILLED)
    
    # Segment individual objects
    segmented_coins = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Create mask for segmentation
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)
        
        segmented_coin = cv2.bitwise_and(image, image, mask=mask)
        segmented_coin = segmented_coin[y:y+h, x:x+w]
        segmented_coins.append(segmented_coin)
    
    return outlined, edges, thresholded_img, blurred, segmented_coins

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {filename}...")
            results = process_image(file_path)
            if results:
                outlined, edges, thresholded_img, blurred, segmented_coins = results
                cv2.imwrite(f'Part1_output/{filename}_outlined.png', outlined)
                cv2.imwrite(f'Part1_output/{filename}_edges.png', edges)
                cv2.imwrite(f'Part1_output/{filename}_thresholded.png', thresholded_img)
                cv2.imwrite(f'Part1_output/{filename}_blurred.png', blurred)
                for i, coin in enumerate(segmented_coins):
                    cv2.imwrite(f'Part1_output/{filename}_segmented_coin_{i+1}.png', coin)

def main():
    folder_path = './Part1_input'
    if not os.path.exists(folder_path):
        print("Invalid folder path. Please provide a valid directory.")
        return
    process_folder(folder_path)

if __name__ == "__main__":
    main()