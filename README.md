# Visual Recognition Assignment: Coin Detection and Image Stitching

## Overview
This project involves two main computer vision tasks:

### Part 1: Coin Detection, Segmentation, and Counting


*Input:* An image containing scattered Indian coins.

### Part 2: Image Stitching to Create a Panorama
*Input:* 3 images with overlapping regions.

## Setup and Execution
1. **Clone the Repository:**
```
git clone <repository-url>
cd <project-folder>
```
2. **Set Up Virtual Environment:**
```
python3 -m venv myenv
source myenv/bin/activate  # For Linux/Mac
myenv\Scripts\activate     # For Windows
```
3. **Install Required Libraries:**
```
pip install -r requirements.txt
```
4. **Run Coin Detection and Segmentation:**
```
python3 Part1.py
```
5. **Run Image Stitching:**
```
python3 Part2.py
```
6. **View Outputs:**
- Outlined Coins in `Part1_output/{filename}_outlined.png`
- Separated coins images in `Part1_output/{filename}_segmented_coin_{i}.png`
- Stitched panorama in `Part2_output/stitched_final_{i}.jpeg`

## Requirements
- Python 3.8+
- OpenCV
- NumPy
- Matplotlib


