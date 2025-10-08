# Krishi-mitra


KrishiMitra is an AI-powered crop recommendation and disease prediction system designed to help farmers make informed decisions about crop selection, planting, and disease management. The project provides personalized suggestions based on location, soil type, season, and weather conditions.

## Features

- **Crop Recommendation:** Suggests the best crops for a given district, soil type, and season.
- **Disease Prediction:** Detects potential crop diseases using machine learning models.
- **Weather-Aware Suggestions:** Provides weekly rainfall forecasts, drought/flood alerts, and seed sowing/harvesting recommendations.
- **Interactive Interface:** User-friendly interface for farmers to input data and get instant recommendations.
- **Data Analysis & Visualization:** Generates graphs and reports for better decision-making.

## Technologies Used

- Python
- Machine Learning (scikit-learn, TensorFlow/PyTorch)
- Streamlit for Web Interface
- Pandas, NumPy, Matplotlib/Seaborn for data processing and visualization
- JSON for data storage

## Installation

1. Clone the repository:  
git clone https://github.com/Shravsz/Krishi-mitra.git

Navigate to the project folder:
cd krishi_mitra


Install required dependencies:
pip install -r requirements.txt

Usage
Run the Streamlit app:
streamlit run app.py


Input your location, soil type, and season to get crop recommendations.

Upload crop images for disease prediction.

Project Structure
krishi_mitra/
├── app.py
├── data/
├── static/
├── templates/
├── users.json
├── schemes_data.json
└── README.md
