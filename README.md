# ChurnShield AI — Customer Churn Predictor

An advanced Flask web application that predicts customer churn using a Random Forest Machine Learning model with explainable AI, interactive analytics dashboards, and a modern SaaS-style UI.

---

# 🚀 Features

## Authentication System
- User Registration
- User Login & Logout
- Secure password hashing using Werkzeug
- Session-based authentication

---

## Dashboard
- Personalized user dashboard
- Prediction statistics
- Recent prediction history
- Fixed sidebar navigation

---

## Customer Churn Prediction
- Real-time churn prediction using Machine Learning
- Random Forest Classifier
- Telecom customer churn analysis
- Probability-based predictions

---

## Explainable AI
- SHAP-based prediction explanations
- Feature importance visualization
- Human-readable churn reasons

---

## Analytics & Visualizations
- Churn probability gauge
- Doughnut chart
- Feature importance bar graph
- Interactive Chart.js visualizations

---

## Prediction History
- SQLite database integration
- Stores all previous predictions
- Timestamp tracking
- Prediction history page

---

## CSV Bulk Prediction
- Upload customer CSV files
- Bulk churn prediction
- Download prediction results

---

## UI / UX
- Modern dark SaaS-style interface
- Responsive design
- Glassmorphism-inspired cards
- Animated UI elements
- Professional dashboard layout

---

# 🛠️ Tech Stack

## Frontend
- HTML
- CSS
- JavaScript
- Chart.js
- Font Awesome

---

## Backend
- Python
- Flask

---

## Machine Learning
- Scikit-learn
- Random Forest Classifier
- SHAP
- Pandas
- NumPy

---

## Database
- SQLite

---

# 📁 Project Structure

```bash
churn_predictor/
│
├── app.py
├── requirements.txt
├── Procfile
├── README.md
│
├── model/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── feature_importances.pkl
│   ├── feature_names.pkl
│   └── train_model.py
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── predict.html
│   ├── result.html
│   ├── history.html
│   ├── upload_csv.html
│   └── csv_result.html
│
├── static/
│   └── css/
│       └── main.css
│
├── instance/
│   └── churn.db
│
└── data/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
⚙️ Installation & Setup
1. Clone Repository
git clone https://github.com/your-username/ChurnShield-AI-Customer-Churn-Predictor.git
cd ChurnShield-AI-Customer-Churn-Predictor
2. Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Linux / Mac
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Train the Model
python model/train_model.py

This generates:

model.pkl
scaler.pkl
encoders.pkl
feature_importances.pkl
5. Run Application
python app.py

Open browser:

http://127.0.0.1:5000
🤖 Machine Learning Model
Algorithm Used
Random Forest Classifier
ML Workflow
Data preprocessing
Label Encoding
Feature Scaling
Model training
Probability prediction
SHAP explainability
Features Used
Gender
Senior Citizen
Partner
Dependents
Tenure
Internet Service
Contract Type
Payment Method
Monthly Charges
Total Charges
📊 Visualizations
Churn vs Retention Doughnut Chart
Feature Importance Bar Graph
Churn Probability Gauge
Prediction Analytics Dashboard
🔐 Security Features
Password hashing
Protected routes
Session authentication
SQLite parameterized queries
🚀 Deployment

This project is deployment-ready for:

Render
Railway
Heroku
📌 Future Improvements
Dark / Light mode toggle
Email notifications
Admin dashboard
PostgreSQL integration
Docker support
FastAPI backend
Real-time analytics
👨‍💻 Author

Pranjal Patil

Made with ❤️ using Flask + Machine Learning