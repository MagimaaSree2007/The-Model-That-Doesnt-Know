
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="The Model That Doesn't Know",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 The Model That Doesn't Know")
st.write(
    "Upload a clothing image to check whether the CNN "
    "recognizes it or rejects it as Unknown."
)

# --------------------------------------------------
# 2. Check that the weights file exists
# --------------------------------------------------
WEIGHTS_PATH = "model.weights.h5"

if not os.path.exists(WEIGHTS_PATH):
    st.error(
        "Model weights file not found! "
        "Place model.weights.h5 in the same folder as app.py."
    )
    st.stop()

# --------------------------------------------------
# 3. Rebuild the CNN and load trained weights
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(5, activation="softmax")
    ])

    model.load_weights(WEIGHTS_PATH)
    return model


try:
    model = load_model()
    st.success("Trained CNN loaded successfully!")
except Exception as error:
    st.error(f"Could not load the model weights: {error}")
    st.stop()

# --------------------------------------------------
# 4. Known classes and confidence threshold
# --------------------------------------------------
class_names = [
    "T-shirt/top",
    "Trouser",
    "Dress",
    "Sandal",
    "Sneaker"
]

threshold = 0.9900

st.caption(
    f"Unknown rejection threshold: {threshold * 100:.2f}%"
)

# --------------------------------------------------
# 5. Upload an image
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a clothing image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Display the original uploaded image
    original_image = Image.open(uploaded_file)
    st.image(
        original_image,
        caption="Uploaded image",
        width=250
    )

    # Convert to grayscale and resize for Fashion-MNIST
    image = original_image.convert("L")
    image = image.resize((28, 28))

    # Normalize pixel values to the range 0 to 1
    image_array = np.array(
        image, dtype=np.float32
    ) / 255.0

    # Shape: (1, 28, 28, 1)
    image_array = np.expand_dims(
        image_array, axis=(0, -1)
    )

    # --------------------------------------------------
    # 6. Predict the image class
    # --------------------------------------------------
    probabilities = model.predict(
        image_array, verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    st.subheader("Prediction results")

    st.metric(
        "Maximum confidence",
        f"{confidence * 100:.2f}%"
    )

    # --------------------------------------------------
    # 7. Accept or reject the prediction
    # --------------------------------------------------
    if confidence < threshold:
        st.error("🚫 Prediction: UNKNOWN")
        st.write(
            "The highest predicted probability is below "
            "the selected threshold."
        )
    else:
        st.success(
            f"✅ Prediction: {class_names[predicted_index]}"
        )
        st.write(
            "The prediction passed the confidence threshold."
        )

    # --------------------------------------------------
    # 8. Display probabilities for all known classes
    # --------------------------------------------------
    st.subheader("Class probabilities")

    for index, name in enumerate(class_names):
        probability = float(probabilities[index])

        st.write(f"**{name}:** {probability * 100:.2f}%")
        st.progress(probability)

    st.info(
        "Note: Confidence-based rejection is not perfect. "
        "An unknown image may still receive a high confidence "
        "score and be incorrectly accepted as a known class."
    )
