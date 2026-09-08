https://screen-habits-ui-production.up.railway.app/



📱 Screen Addiction Prediction Using LightGBM

A Machine Learning classification project that predicts screen addiction using user-related behavioral and lifestyle features.

The project uses LightGBM (Light Gradient Boosting Machine) to learn patterns from the dataset and generate predictions based on new input data.

🎯 Project Objective

The main objective of this project is to build a Machine Learning model capable of predicting the likelihood of screen addiction from relevant user data.

This project demonstrates a complete Machine Learning workflow, from data preprocessing and exploratory data analysis to model training, evaluation, and prediction.

«⚠️ Disclaimer: This project is created for educational and Machine Learning purposes. The predictions should not be considered a medical or psychological diagnosis.»

🤖 Machine Learning Model

Algorithm: LightGBM Classifier

LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It is designed to provide efficient training and strong performance, particularly on structured/tabular datasets.

🔄 Machine Learning Workflow

Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis (EDA)
   ↓
Data Preprocessing
   ↓
Feature Engineering / Selection
   ↓
Train-Test Split
   ↓
LightGBM Model Training
   ↓
Model Evaluation
   ↓
Screen Addiction Prediction

🛠️ Technologies & Libraries

- 🐍 Python
- 🐼 Pandas — Data manipulation and preprocessing
- 🔢 NumPy — Numerical operations
- 📊 Matplotlib — Data visualization
- 📈 Seaborn — Exploratory data visualization
- 🤖 LightGBM — Machine Learning model
- 📚 Scikit-learn — Data splitting and model evaluation
- 💻 Jupyter Notebook / VS Code — Development environment
- 🎨 Streamlit — Interactive web application (if deployed)

📊 Model Evaluation

The LightGBM model was evaluated using classification metrics including:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics provide a better understanding of how well the model performs rather than relying only on accuracy.

🎯 Prediction

After training, the model can take new input data and predict the corresponding screen addiction class.

The trained model can also be integrated into a web-based application, allowing users to enter their information and receive a prediction.

📂 Project Structure

Screen-Addiction-Prediction/
│
├── dataset/
│   └── dataset.csv
│
├── notebook/
│   └── screen_addiction_prediction.ipynb
│
├── model/
│   └── lightgbm_model.pkl
│
├── app.py
├── requirements.txt
└── README.md

🚀 Future Improvements

- 🔧 Hyperparameter tuning
- 🔄 Cross-validation
- 📊 Feature importance analysis
- 🧠 Compare LightGBM with other classification algorithms
- ⚖️ Handle class imbalance if required
- 🌐 Deploy the model as a web application
- 📈 Improve the model using additional real-world data

👨‍💻 About the Project

This project is part of my journey into Data Science and Machine Learning, where I am building practical projects to strengthen my skills in data preprocessing, exploratory data analysis, model training, evaluation, and deployment.

⭐ If you find this project useful, consider giving the repository a star!
