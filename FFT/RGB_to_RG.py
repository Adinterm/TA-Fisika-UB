import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def calculate_sum_red_green(image_path):
    image = cv2.imread(image_path)
    red_channel = image[:, :, 2]
    green_channel = image[:, :, 1]
    sum_red_green = np.mean(red_channel + green_channel)
    return sum_red_green

def plot_sum_red_green(sum_red_green_values):
    plt.figure()
    plt.title("Sum of Red and Green Values Over Time")
    plt.xlabel("Image Index")
    plt.ylabel("Sum of Red and Green Values")
    plt.plot(sum_red_green_values, marker='o')
    plt.show()

def save_to_excel(image_names, sum_red_green_values, file_name="sum_red_green.xlsx"):
    df = pd.DataFrame({'Image Name': image_names, 'Sum of Red and Green': sum_red_green_values})
    df.to_excel(file_name, index=False)
    print(f"Saved sum of red and green values to {file_name}")

def main(image_directory):
    sum_red_green_values = []
    image_names = []

    for image_name in sorted(os.listdir(image_directory)):
        image_path = os.path.join(image_directory, image_name)
        sum_red_green = calculate_sum_red_green(image_path)
        sum_red_green_values.append(sum_red_green)
        image_names.append(image_name)
        print(f"Sum of red and green for {image_name}: {sum_red_green}")

    plot_sum_red_green(sum_red_green_values)
    save_to_excel(image_names, sum_red_green_values)

if __name__ == "__main__":
    # Directory containing the images
    folder_name = "Desktop/images"
    image_directory = os.path.join(os.getcwd(), folder_name)
    main(image_directory)