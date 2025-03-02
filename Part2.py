import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def read_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {file_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def get_sift_features(gray_img):
    if not hasattr(cv2, 'SIFT_create'):
        raise AttributeError("SIFT is not available in your OpenCV version")
    sift_detector = cv2.SIFT_create()
    return sift_detector.detectAndCompute(gray_img, None)

# def show_keypoints(gray_img, color_img, keypoints):
#     kp_img = cv2.drawKeypoints(gray_img, keypoints, color_img.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     plt.figure(figsize=(8, 8))
#     plt.imshow(kp_img)
#     plt.axis('off')
#     plt.show()
#     return kp_img

def find_keypoint_matches(kp1, des1, kp2, des2, ratio_thresh=0.5):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    return np.array([[kp1[m.queryIdx].pt + kp2[m.trainIdx].pt] for m in good_matches]).reshape(-1, 4)

# def plot_matches(matches, img1, img2):
#     combined = np.concatenate((img1, img2), axis=1)
#     shift = img1.shape[1]
#     plt.figure()
#     plt.imshow(combined)
#     plt.plot(matches[:, 0], matches[:, 1], 'xr')
#     plt.plot(matches[:, 2] + shift, matches[:, 3], 'xr')
#     plt.plot([matches[:, 0], matches[:, 2] + shift], [matches[:, 1], matches[:, 3]], 'r', linewidth=0.1)
#     plt.axis('off')
#     plt.show()

def merge_images(img_base, img_target, matches):
    pts1, pts2 = matches[:, :2], matches[:, 2:]
    transform_matrix, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    merged = cv2.warpPerspective(img_target, transform_matrix, (img_base.shape[1] + img_target.shape[1], img_base.shape[0]))
    merged[0:img_base.shape[0], 0:img_base.shape[1]] = img_base
    return merged

def process_and_stitch(img1_path, img2_path):
    img1_gray, img1_original, img1_color = read_image(img1_path)
    img2_gray, img2_original, img2_color = read_image(img2_path)

    kp1, des1 = get_sift_features(img1_gray)
    kp2, des2 = get_sift_features(img2_gray)
    
    # show_keypoints(img1_gray, img1_color, kp1)
    # show_keypoints(img2_gray, img2_color, kp2)
    
    matches = find_keypoint_matches(kp1, des1, kp2, des2)    
    return merge_images(img1_color, img2_color, matches)

def stitch_images_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Identify images dynamically based on pattern
    left_images = sorted([f for f in os.listdir(input_folder) if f.startswith("left_")])
    centre_images = sorted([f for f in os.listdir(input_folder) if f.startswith("centre_")])
    right_images = sorted([f for f in os.listdir(input_folder) if f.startswith("right_")])
    
    if not (left_images and centre_images and right_images):
        raise ValueError("Could not find left, centre, and right images in the given folder.")
    
    for i in range(len(left_images)):
        left_path = os.path.join(input_folder, left_images[i])
        centre_path = os.path.join(input_folder, centre_images[i])
        right_path = os.path.join(input_folder, right_images[i])

        print(f"Processing: {left_path}, {centre_path}, {right_path}")

        stitched_part1 = process_and_stitch(centre_path, right_path)
        stitched1_path = os.path.join(input_folder, f"stitched1_{i+1}.jpeg")
        cv2.imwrite(stitched1_path, cv2.cvtColor(stitched_part1, cv2.COLOR_RGB2BGR))

        final_stitched = process_and_stitch(left_path, stitched1_path)

        # Crop black borders
        gray = cv2.cvtColor(final_stitched, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cropped_image = final_stitched[y:y+h, x:x+w]
            final_output_path = os.path.join(output_folder, f"stitched_final_{i+1}.jpeg")
            cv2.imwrite(final_output_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
            print(f"Saved: {final_output_path}")
        else:
            print("No contours found in the stitched image.")

def main():
    stitch_images_in_folder('Part2_input', 'Part2_output')

if __name__ == "__main__":
    main()