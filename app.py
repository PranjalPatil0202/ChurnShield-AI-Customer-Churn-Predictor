import os
import json
import pickle
import sqlite3
import hashlib
import secrets
from datetime import datetime
from functools import wraps
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify, g)
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
DATABASE = os.path.join(app.instance_path, 'churn.db')
os.makedirs(app.instance_path, exist_ok=True)

# ── Load ML artifacts ──────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

with open(os.path.join(MODEL_DIR, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'encoders.pkl'), 'rb') as f:
    encoders = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'feature_importances.pkl'), 'rb') as f:
    feature_importances = pickle.load(f)

# ── DB helpers ─────────────────────────────────────────────────────────────────
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_db(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                customer_id TEXT,
                input_data TEXT NOT NULL,
                churn_result INTEGER NOT NULL,
                churn_probability REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        ''')
        db.commit()

init_db()

# ── Auth decorator ─────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def current_user():
    if 'user_id' not in session:
        return None
    db = get_db()
    return db.execute('SELECT * FROM users WHERE id=?', (session['user_id'],)).fetchone()

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        errors = []
        if not full_name or len(full_name) < 2:
            errors.append('Full name must be at least 2 characters.')
        if not email or '@' not in email:
            errors.append('Enter a valid email address.')
        if len(password) < 6:
            errors.append('Password must be at least 6 characters.')
        if password != confirm:
            errors.append('Passwords do not match.')
        db = get_db()
        if db.execute('SELECT id FROM users WHERE email=?', (email,)).fetchone():
            errors.append('Email already registered.')
        if errors:
            for e in errors:
                flash(e, 'danger')
            return render_template('register.html', full_name=full_name, email=email)
        pw_hash = generate_password_hash(password)
        db.execute('INSERT INTO users (full_name, email, password_hash) VALUES (?,?,?)',
                   (full_name, email, pw_hash))
        db.commit()
        flash('Account created! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['user_name'] = user['full_name']
            flash(f'Welcome back, {user["full_name"].split()[0]}!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db()
    user = current_user()
    total_preds = db.execute('SELECT COUNT(*) FROM predictions WHERE user_id=?',
                             (session['user_id'],)).fetchone()[0]
    last_pred = db.execute(
        'SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 1',
        (session['user_id'],)).fetchone()
    recent = db.execute(
        'SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 5',
        (session['user_id'],)).fetchall()
    return render_template('dashboard.html', user=user, total_preds=total_preds,
                           last_pred=last_pred, recent=recent)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    user = current_user()
    if request.method == 'POST':
        try:
            customer_id = request.form.get('customer_id', 'N/A')
            gender = request.form['gender']
            age = int(request.form['age'])
            tenure = int(request.form['tenure'])
            monthly_charges = float(request.form['monthly_charges'])
            total_charges = float(request.form['total_charges'])
            contract_type = request.form['contract_type']
            internet_service = request.form['internet_service']
            payment_method = request.form['payment_method']
            support_calls = int(request.form['support_calls'])
            senior_citizen = int(request.form.get('senior_citizen', 0))

            gender_enc = encoders['gender'].transform([gender])[0]
            contract_enc = encoders['contract'].transform([contract_type])[0]
            internet_enc = encoders['internet'].transform([internet_service])[0]
            payment_enc = encoders['payment'].transform([payment_method])[0]

            features = np.array([[gender_enc, age, tenure, monthly_charges,
                                  total_charges, contract_enc, internet_enc,
                                  payment_enc, support_calls, senior_citizen]])
            features_scaled = scaler.transform(features)
            pred = int(model.predict(features_scaled)[0])
            prob = float(model.predict_proba(features_scaled)[0][1])

            input_data = {
                'customer_id': customer_id, 'gender': gender, 'age': age,
                'tenure': tenure, 'monthly_charges': monthly_charges,
                'total_charges': total_charges, 'contract_type': contract_type,
                'internet_service': internet_service, 'payment_method': payment_method,
                'support_calls': support_calls, 'senior_citizen': senior_citizen
            }

            db = get_db()
            db.execute(
                '''INSERT INTO predictions (user_id, customer_id, input_data,
                   churn_result, churn_probability) VALUES (?,?,?,?,?)''',
                (session['user_id'], customer_id, json.dumps(input_data), pred, prob))
            db.commit()

            # Build explanations
            explanations = []
            if tenure < 12:
                explanations.append({'icon': '📅', 'text': 'Very short tenure (< 12 months) — new customers churn more often.'})
            if monthly_charges > 1500:
                explanations.append({'icon': '💸', 'text': f'High monthly charges (₹{monthly_charges:.0f}) increase churn risk.'})
            if support_calls > 5:
                explanations.append({'icon': '📞', 'text': f'{support_calls} support calls indicate recurring issues.'})
            if contract_type == 'Monthly':
                explanations.append({'icon': '📋', 'text': 'Monthly contracts have lower commitment — easier to leave.'})
            if internet_service == 'Fiber':
                explanations.append({'icon': '🌐', 'text': 'Fiber users tend to be more demanding and price-sensitive.'})
            if senior_citizen:
                explanations.append({'icon': '👴', 'text': 'Senior citizens are a higher-risk demographic.'})
            if not explanations:
                explanations.append({'icon': '✅', 'text': 'Customer profile looks stable with low churn risk.'})

            sorted_fi = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            fi_labels = [k for k, v in sorted_fi]
            fi_values = [round(v * 100, 2) for k, v in sorted_fi]

            return render_template('result.html', user=user,
                                   pred=pred, prob=round(prob * 100, 1),
                                   input_data=input_data,
                                   explanations=explanations,
                                   fi_labels=json.dumps(fi_labels),
                                   fi_values=json.dumps(fi_values))
        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'danger')
    return render_template('predict.html', user=user)

@app.route('/history')
@login_required
def history():
    user = current_user()
    db = get_db()
    preds = db.execute(
        'SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC',
        (session['user_id'],)).fetchall()
    parsed = []
    for p in preds:
        d = dict(p)
        d['input_data'] = json.loads(d['input_data'])
        parsed.append(d)
    return render_template('history.html', user=user, predictions=parsed)

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
