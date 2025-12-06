# Overview
This project is a Dog Breed Recognition System that identifies the breed of a dog from an uploaded image using a trained CNN model (ResNet).
It also provides:

# Detailed breed information (loaded from JSON)
# AI Chatbot powered by Gemini
# Beautiful dog-themed Streamlit UI
# Animated popup chatbot on UI
# Works with 120 Dog Breeds

# Features:
Dog Breed Classifier
Upload an image (jpg, png, webp, avif)
Model predicts the correct breed
Shows confidence score
“Know More” button reveals breed details
🤖 AI Dog Chatbot
Asks questions about dogs (height, lifespan, temperament, etc.)
Powered by Gemini 2.5 Flash
Aesthetic, animated popup design
Streamlit Web App
Interactive & responsive

# Tech Stack:
Component	Technology
Model-TensorFlow / Keras
UI-	Streamlit
Chatbot-Google Gemini API
Breed Data-	JSON file
Deployment-	Streamlit Cloud
Images-	Pillow (PIL)
Misc-	NumPy, Python 3.11

# Folder Structure
.
├── app.py
├── dog_breed_resnet.keras
├── class_indices.json
├── 120_breeds_new.json
├── requirements.txt
└── README.md
