import numpy as np
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.image import img_to_array
from skimage import transform

Models = ['Model CNN.keras', 'Model EffNET.keras']

# Class names for each model
class_names_all = [
    ['Tomato___bacterial_spot', 'Tomato___early_blight', 'Tomato___healthy', 'Tomato___late_blight',
     'Tomato___leaf_curl', 'Tomato___leaf_mold', 'Tomato___mosaic_virus', 'Tomato___septoria_leaf_spot',
     'Tomato___spider_mites', 'Tomato___target_spot'],  # CNN

    ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
     'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
     'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
     'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']  # EfficientNet
]

def get_model(modelNo):
    model_path = "./static/Models/" + Models[modelNo]
    model = load_model(model_path)
    return model

def predict(image_data, modelNo):
    loaded_model = get_model(modelNo)
    img = img_to_array(image_data)
    np_image = transform.resize(img, (224, 224, 3))
    image4 = np.expand_dims(np_image, axis=0)
    prediction = loaded_model.predict(image4)
    class_names = class_names_all[modelNo]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    return predicted_class, confidence
