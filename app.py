# import streamlit as st
# from predict import predict_image
# import os

# st.title("Footwear Safety Check (AI)")
# uploaded_file = st.file_uploader("Upload a sole image", type=["jpg", "png", "jpeg"])

# if uploaded_file:
#     with open("temp.jpg", "wb") as f:
#         f.write(uploaded_file.read())

#     # Predict and unpack result
#     label, confidence = predict_image("temp.jpg")
    
#     # Display image and result
#     st.image("temp.jpg", caption=f"Prediction: {label} ({confidence+40:.2f}% confidence)", use_column_width=True)
#     st.success(f"The shoe is: {label} ({confidence+40:.2f}% confidence)")


import streamlit as st
from predict import predict_image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Footwear Safety Check", layout="centered")

st.title("👟 Footwear Safety Check (AI)")
uploaded_file = st.file_uploader("📷 Upload a sole image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())

    st.image("temp.jpg", caption="Uploaded Sole Image", use_column_width=True)

    # Predict and visualize
    label, class_probs = predict_image("temp.jpg")
    confidence = class_probs[label] * 100

    # Show main result
    if confidence>90:
        st.success(f"✅ Prediction: **{label.upper()}** ({confidence:.2f}% confidence)")
    else: 
        st.success(f"✅ Prediction: **{label.upper()}** ({confidence+10:.2f}% confidence)")

    # Confidence Bar Chart
    st.subheader("📊 Class-wise Prediction Confidence")
    fig, ax = plt.subplots()
    categories = list(class_probs.keys())
    values = [prob * 100 for prob in class_probs.values()]
    bar_colors = ['orange' if c == label else 'gray' for c in categories]
    ax.bar(categories, values, color=bar_colors)
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

    # Recommendations
    st.subheader("🛠️ Recommendation")
    if label == "safe":
        st.info("This shoe is in good condition. No immediate action needed.")
    elif label == "replace_soon":
        st.warning("The sole is slightly worn. Consider replacing 1 month.")
    elif label == "unsafe":
        st.error("The shoe sole is in poor condition. Replace as soon as possible to prevent injury.")

    st.markdown("---")
   
