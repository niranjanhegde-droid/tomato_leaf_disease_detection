import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from skimage import transform

# Model names
Models = ['Model CNN.keras', 'Model EffNET.keras']

# Class names for each model
class_names_all = [
    # CNN classes
    [
        'Tomato___bacterial_spot', 'Tomato___early_blight', 'Tomato___healthy', 'Tomato___late_blight',
        'Tomato___leaf_curl', 'Tomato___leaf_mold', 'Tomato___mosaic_virus', 'Tomato___septoria_leaf_spot',
        'Tomato___spider_mites', 'Tomato___target_spot'
    ],

    # EfficientNet classes
    [
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
    ]
]

def get_model(modelNo):
    model_path = "./static/Models/" + Models[modelNo]
    model = load_model(model_path, compile=False)
    return model

def predict(image_data, modelNo):
    # Load selected model
    loaded_model = get_model(modelNo)

    # Convert image to array
    img = img_to_array(image_data)

    # Resize to model input (224x224)
    np_image = transform.resize(img, (224, 224, 3), anti_aliasing=True)

    # Convert to float32 for compatibility
    np_image = np.array(np_image, dtype=np.float32)

    # Normalize if needed
    np_image = np_image / 255.0

    # Add batch dimension
    image4 = np.expand_dims(np_image, axis=0)

    # Prediction
    prediction = loaded_model.predict(image4)

    # Get class names
    class_names = class_names_all[modelNo]

    # Result
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    return predicted_class, confidence
