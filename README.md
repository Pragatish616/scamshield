# 🛡️ ScamShield -- AI Powered Scam Detection API

ScamShield is a **Machine Learning powered API service** designed to
detect scam or phishing messages in real time. The system analyzes user
input and predicts whether the content is **scam or legitimate** using a
trained ML model.

This project demonstrates an **end-to-end AI pipeline**, including data
preprocessing, model training, API development, and cloud deployment.

🌐 **Live API Documentation:**\
https://scamshield-14i0.onrender.com/docs

------------------------------------------------------------------------

# 📌 Problem Statement

Online scams, phishing messages, and fraudulent links are increasing
rapidly across messaging platforms, emails, and websites.

Many users struggle to identify malicious content, leading to:

-   Financial fraud
-   Identity theft
-   Data breaches

There is a need for an **automated system that can detect scams
instantly and warn users before they fall victim.**

------------------------------------------------------------------------

# 🎯 Solution

ScamShield uses **Machine Learning and Natural Language Processing
(NLP)** to analyze suspicious text and classify it as:

-   **Scam**
-   **Not Scam**

The model processes the input text, extracts meaningful features, and
predicts the likelihood of a scam.

The system is exposed through a **REST API**, allowing integration into:

-   Mobile applications
-   Messaging platforms
-   Browser extensions
-   Email security tools

------------------------------------------------------------------------

# ⚙️ System Architecture

User Input (Message / URL) ↓ FastAPI Backend ↓ Text Preprocessing ↓
Feature Extraction (NLP) ↓ Machine Learning Model ↓ Prediction (Scam /
Safe) ↓ JSON Response

------------------------------------------------------------------------

# 🧠 Machine Learning Pipeline

### 1️⃣ Data Collection

A dataset containing examples of scam and legitimate messages was used
for training.

### 2️⃣ Data Preprocessing

Text cleaning steps include: - Lowercasing - Removing special
characters - Tokenization - Stopword removal

### 3️⃣ Feature Extraction

Text data is converted into numerical features using **TF-IDF
vectorization**.

### 4️⃣ Model Training

The processed data is used to train a classification model capable of
distinguishing scam patterns.

### 5️⃣ Prediction

The trained model predicts whether new input text is **scam or safe**.

------------------------------------------------------------------------

# 🚀 API Endpoints

### Base URL

https://scamshield-14i0.onrender.com

### Predict Scam

Endpoint: POST /predict

Request Example:

{ "text": "Your bank account will be blocked. Click the link
immediately." }

Response Example:

{ "prediction": "Scam", "confidence": 0.94 }

------------------------------------------------------------------------

# 🛠️ Tech Stack

  Technology     Purpose
  -------------- ---------------------
  Python         Backend development
  FastAPI        API framework
  Scikit-learn   Machine learning
  NLP            Text processing
  Render         Cloud deployment
  Swagger UI     API testing

------------------------------------------------------------------------

# ☁️ Deployment

The API is deployed on **Render**, enabling public access to the model
for testing and integration.

Deployment pipeline:

ML Model → FastAPI Server → Render Cloud → Public API

------------------------------------------------------------------------

# 🔍 Use Cases

ScamShield can be integrated into multiple platforms:

-   📱 Mobile security applications
-   🌐 Browser extensions for phishing detection
-   💬 Messaging platforms to filter scam messages
-   📧 Email spam filtering systems
-   🏦 Banking applications for fraud detection

------------------------------------------------------------------------

# 🧪 Testing the API

You can test the API directly using the Swagger interface:

https://scamshield-14i0.onrender.com/docs

Simply enter a text message and view the prediction response.

------------------------------------------------------------------------

# 🔮 Future Improvements

Planned enhancements include:

-   URL phishing detection
-   Real-time scam risk scoring
-   Deep learning based NLP models
-   Browser extension integration
-   Multi-language scam detection

------------------------------------------------------------------------

## 👨‍💻 Contributors

This project was developed by:

1.  **Gokul Nath S**
2.  **Pragatish N**
3.  **Jyotish N**

------------------------------------------------------------------------

# 📜 License

This project is intended for **educational and research purposes**.
