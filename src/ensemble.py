import numpy as np
import torch
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image

weights = {
    "yolo": 0.3,
    "resnet": 0.5,
    "vit": 0.05,
    "cnn": 0.15
}


def ensemble_predict(image_path, yolo_model, resnet_model, vit_model, cnn_model, transform, device, weights):
    # --- YOLO ---
    yolo_preds = yolo_model.predict(image_path)
    yolo_probs = yolo_preds[0].probs.data.cpu().numpy()

    # --- ResNet ---
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    resnet_probs = torch.softmax(resnet_model(img_t), dim=1).detach().cpu().numpy()[0]

    # --- ViT ---
    vit_probs = torch.softmax(vit_model(img_t), dim=1).detach().cpu().numpy()[0]


    # --- CNN ---
    img_cnn = keras_image.load_img(image_path, target_size=(150, 220))
    img_cnn = keras_image.img_to_array(img_cnn) / 255.0
    img_cnn = np.expand_dims(img_cnn, axis=0)
    cnn_probs = cnn_model.predict(img_cnn)[0]

    # --- Weighted Voting ---
    final_probs = (
        weights["yolo"] * yolo_probs +
        weights["resnet"] * resnet_probs +
        weights["vit"] * vit_probs +
        weights["cnn"] * cnn_probs
    )
    final_class = np.argmax(final_probs)
    return final_class, final_probs
