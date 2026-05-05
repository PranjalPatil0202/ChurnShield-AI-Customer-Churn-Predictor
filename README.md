# ChurnShield AI — Customer Churn Predictor

A production-ready Flask web application that predicts customer churn using a Random Forest ML model with explainable AI, interactive Chart.js visualizations, and a premium dark-themed SaaS UI.

---

## 🚀 Features

- **Authentication** — Register, Login, Logout with Werkzeug password hashing + session management
- **Dashboard** — Personalized welcome, prediction stats, recent history table
- **Churn Prediction Form** — 10 Indian-context features (UPI, ₹ pricing, etc.)
- **Result Page** — Churn verdict, probability gauge, pie chart, feature importance bar chart, customer profile chart
- **Explainable AI** — Human-readable reasons for every prediction
- **Prediction History** — All past predictions stored in SQLite
- **Premium UI** — Dark SaaS design, Syne + DM Sans fonts, glassmorphic cards, animations
- **Error Pages** — Custom 404 and 500 handlers
- **Deployment Ready** — Procfile, requirements.txt, .env support

---

## 📁 Project Structure

```
churn_predictor/
├── app.py                    # Main Flask application
├── requirements.txt
├── Procfile                  # For Render/Railway/Heroku
├── .env.example
├── model/
│   ├── train_model.py        # ML training script
│   ├── model.pkl             # Trained Random Forest model
│   ├── scaler.pkl            # StandardScaler
│   ├── encoders.pkl          # LabelEncoders
│   ├── feature_importances.pkl
│   └── churn_dataset.csv     # Generated training data
├── templates/
│   ├── base.html
│   ├── index.html            # Landing page
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── predict.html
│   ├── result.html
│   ├── history.html
│   ├── 404.html
│   └── 500.html
├── static/
│   ├── css/main.css
│   └── js/main.js
└── instance/
    └── churn.db              # SQLite database (auto-created)
```

---

## ⚙️ Local Setup

### 1. Clone / Extract the project
```bash
cd churn_predictor
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the ML model (only needed once)
```bash
python model/train_model.py
```
This generates `model.pkl`, `scaler.pkl`, `encoders.pkl`, and `feature_importances.pkl`.

### 5. Set environment variables
```bash
cp .env.example .env
# Edit .env and set a strong SECRET_KEY
```

### 6. Run the app
```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 🌐 Deployment (Render / Railway)

1. Push the project to a GitHub repo
2. Connect to Render / Railway
3. Set environment variable: `SECRET_KEY=your-secret`
4. Build command: `pip install -r requirements.txt && python model/train_model.py`
5. Start command: `gunicorn app:app`

---

## 🤖 ML Model Details

- **Algorithm**: Random Forest Classifier (100 estimators, max_depth=10)
- **Features**: Gender, Age, Tenure, Monthly Charges, Total Charges, Contract Type, Internet Service, Payment Method, Support Calls, Senior Citizen
- **Preprocessing**: LabelEncoding for categoricals, StandardScaler for numericals
- **Dataset**: 2000 synthetic Indian telecom customers
- **Accuracy**: ~69% (balanced dataset with realistic churn patterns)

---

## 📊 Visualizations (Chart.js)

- **Doughnut Chart** — Churn probability vs retention split
- **Horizontal Bar Chart** — Feature importance from Random Forest
- **Bar Chart** — Customer profile key metrics (tenure, charges, calls, age)
- **Probability Gauge** — CSS conic-gradient animated gauge

---

## 🔐 Security Notes

- Passwords hashed with Werkzeug PBKDF2-SHA256
- Sessions protected by Flask secret key
- All routes requiring auth use `@login_required` decorator
- SQL injection protected via parameterized SQLite queries
- Never commit `.env` to version control

---

Made with ❤️ for Indian Businesses | Flask + scikit-learn + Chart.js
