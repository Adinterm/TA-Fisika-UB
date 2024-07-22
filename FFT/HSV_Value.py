import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def rgb_to_hsv(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def calculate_average_hue(hsv_image):
    hue_channel = hsv_image[:, :, 0]
    # Consider hue values in the range from red to orange (0 to 30 in HSV space)
    mask = (hue_channel >= 0) & (hue_channel <= 30)
    average_hue = np.mean(hue_channel[mask])
    return average_hue

def plot_average_hues(average_hues):
    plt.figure()
    plt.title("Average Hue Values Over Time")
    plt.xlabel("Image Index")
    plt.ylabel("Average Hue Value (Red to Orange Range)")
    plt.plot(average_hues, marker='o')
    plt.show()

def save_to_excel(image_names, average_hues, file_name="average_hues.xlsx"):
    df = pd.DataFrame({'Image Name': image_names, 'Average Hue': average_hues})
    df.to_excel(file_name, index=False)
    print(f"Saved average hues to {file_name}")

def main(image_directory):
    average_hues = []
    image_names = []

    for image_name in sorted(os.listdir(image_directory)):
        image_path = os.path.join(image_directory, image_name)
        hsv_image = rgb_to_hsv(image_path)
        average_hue = calculate_average_hue(hsv_image)
        average_hues.append(average_hue)
        image_names.append(image_name)
        print(f"Average hue for {image_name}: {average_hue}")

    plot_average_hues(average_hues)
    save_to_excel(image_names, average_hues)

if __name__ == "__main__":
    # Directory containing the images
    folder_name = "Desktop/images"
    image_directory = os.path.join(os.getcwd(), folder_name)
    main(image_directory)