import pydicom
import numpy as np
from PIL import Image
import os

input_folder = "dicom_images"
output_folder = "converted_images"
os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith(".dcm"):
        ds = pydicom.dcmread(os.path.join(input_folder, file_name))
        img_array = ds.pixel_array.astype(float)
        img_array = (np.maximum(img_array, 0) / img_array.max()) * 255.0
        img_array = np.uint8(img_array)
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_folder, file_name.replace(".dcm", ".png")))
        print(f"Converted: {file_name}")
