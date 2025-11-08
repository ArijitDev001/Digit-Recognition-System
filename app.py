import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import keras
from PIL import Image, ImageOps
import io
import cv2

# 🔧 Patch for missing 'softmax_v2' activation
tf.keras.utils.get_custom_objects()['softmax_v2'] = tf.nn.softmax

# Load your trained model
@st.cache_resource
def load_mnist_model():
    model = load_model(r"E:\Project\MNIST Digit recognition\Digit_Recognition_trained_model(0.3)(0.4).h5", compile=False)
    return model

model = load_mnist_model()

#print(model.input_shape)

# Streamlit page config
st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title("🧠 Handwritten Digit Recognition")
st.write("Draw a digit (0–9) below and click **Predict**.")

# Drawing canvas
from streamlit_drawable_canvas import st_canvas

canvas_result = st_canvas(
    fill_color="#000000",  # Black ink
    stroke_width=10,
    stroke_color="#FFFFFF",  # White stroke
    background_color="#000000",  # Black background
    width=300,
    height=300,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict button
if st.button("🔮 Predict"):
    if canvas_result.image_data is not None:
        # Convert image to 28x28 grayscale
        #img = canvas_result.image_data
        #img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGBA2GRAY)
        #img = cv2.resize(img, (28, 28))
        #img = img / 255.0
        #img = np.expand_dims(img, axis=(0, -1))  # shape (1,28,28,1)
        # Convert canvas image to 8-bit grayscale
        img = canvas_result.image_data.astype("uint8")
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        # Invert colors (black digit → white background)
        gray = cv2.bitwise_not(gray)

        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

        # Normalize for model input
        norm = resized / 255.0
        input_img = norm.reshape(1, 28, 28, 1)

        # Show both original & resized images
        st.subheader("🖼️ Processed Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Drawing", use_container_width=True)
        with col2:
            st.image(resized, caption="Resized 28×28 Image", use_container_width=True)

        norm = resized / 255.0
        input_img = norm.reshape(1, 28, 28).astype("float32")  # ✅ Flatten to match model input

        # Predict
        pred = np.argmax(model.predict(input_img),axis=1)[0]
        st.success(f"✨ Predicted Digit: **{pred}**")
    else:
        st.warning("Please draw a digit before predicting!")

# Clear instructions
st.caption("Tip: Use your mouse or touchscreen to draw a digit, then click Predict.")



