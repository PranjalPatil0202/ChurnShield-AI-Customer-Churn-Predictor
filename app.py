import os
import json
import pickle
import sqlite3
import secrets
from functools import wraps

import numpy as np
import pandas as pd
import shap

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    g,
    send_file
)

from werkzeug.security import generate_password_hash, check_password_hash


# =========================================================
# FLASK APP
# =========================================================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

DATABASE = os.path.join(app.instance_path, "churn.db")
os.makedirs(app.instance_path, exist_ok=True)


# =========================================================
# LOAD MODEL FILES
# =========================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

model = pickle.load(open(os.path.join(MODEL_DIR, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))
encoders = pickle.load(open(os.path.join(MODEL_DIR, "encoders.pkl"), "rb"))
feature_importances = pickle.load(
    open(os.path.join(MODEL_DIR, "feature_importances.pkl"), "rb")
)
feature_names = pickle.load(
    open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb")
)

# SHAP EXPLAINER
explainer = shap.TreeExplainer(model)


# =========================================================
# DATABASE
# =========================================================
def get_db():
    db = getattr(g, "_database", None)

    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row

    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)

    if db is not None:
        db.close()


def init_db():
    db = get_db()

    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT,
            email TEXT UNIQUE,
            password_hash TEXT
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            customer_id TEXT,
            input_data TEXT,
            churn_result INTEGER,
            churn_probability REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    db.commit()


with app.app_context():
    init_db()


# =========================================================
# AUTH
# =========================================================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):

        if "user_id" not in session:
            flash("Please login first", "warning")
            return redirect(url_for("login"))

        return f(*args, **kwargs)

    return decorated


def current_user():

    if "user_id" not in session:
        return None

    db = get_db()

    return db.execute(
        "SELECT * FROM users WHERE id=?",
        (session["user_id"],)
    ).fetchone()


# =========================================================
# PREPROCESS INPUT
# =========================================================
def preprocess_input(form_data):

    processed = []

    for feature in feature_names:

        value = form_data.get(feature)

        # Missing values
        if value is None or value == "":
            value = "No"

        # Categorical
        if feature in encoders:

            try:
                value = encoders[feature].transform([str(value)])[0]

            except:
                value = 0

        # Numerical
        else:

            try:
                value = float(value)

            except:
                value = 0.0

        processed.append(value)

    return np.array(processed).reshape(1, -1)


# =========================================================
# HOME
# =========================================================
@app.route("/")
def index():

    # Clear old session
    session.clear()

    # Show landing page
    return render_template(
        "index.html"
    )

# =========================================================
# REGISTER
# =========================================================
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        full_name = request.form["full_name"]
        email = request.form["email"]
        password = request.form["password"]

        hashed_password = generate_password_hash(password)

        db = get_db()

        try:

            db.execute(
                """
                INSERT INTO users(
                    full_name,
                    email,
                    password_hash
                )
                VALUES(?,?,?)
                """,
                (
                    full_name,
                    email,
                    hashed_password
                )
            )

            db.commit()

            flash("Account created successfully", "success")

            return redirect(url_for("login"))

        except:
            flash("Email already exists", "danger")

    return render_template("register.html")


# =========================================================
# LOGIN
# =========================================================
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        db = get_db()

        user = db.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        ).fetchone()

        if user and check_password_hash(
            user["password_hash"],
            password
        ):

            session["user_id"] = user["id"]
            session["user_name"] = user["full_name"]

            return redirect(url_for("dashboard"))

        flash("Invalid credentials", "danger")

    return render_template("login.html")


# =========================================================
# LOGOUT
# =========================================================
@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("index"))


# =========================================================
# DASHBOARD
# =========================================================
@app.route("/dashboard")
@login_required
def dashboard():

    db = get_db()

    user = current_user()

    predictions = db.execute(
        """
        SELECT *
        FROM predictions
        WHERE user_id=?
        ORDER BY id DESC
        """,
        (session["user_id"],)
    ).fetchall()

    total_preds = len(predictions)

    last_pred = predictions[0] if predictions else None

    recent = predictions[:5]

    return render_template(
        "dashboard.html",
        user=user,
        predictions=predictions,
        total_preds=total_preds,
        last_pred=last_pred,
        recent=recent
    )


# =========================================================
# SINGLE PREDICTION
# =========================================================
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():

    user = current_user()

    if request.method == "POST":

        try:

            form_data = request.form.to_dict()

            customer_id = form_data.get(
                "customer_id",
                "N/A"
            )

            # =====================================
            # ML PIPELINE
            # =====================================
            input_array = preprocess_input(form_data)

            input_scaled = scaler.transform(input_array)

            pred = int(
                model.predict(input_scaled)[0]
            )

            prob = float(
                model.predict_proba(input_scaled)[0][1]
            )

            # =====================================
            # SHAP EXPLANATIONS
            # =====================================
            shap_values = explainer.shap_values(input_scaled)

            try:
                shap_vals = shap_values[1][0]

            except:
                shap_vals = shap_values[0]

            feature_contrib = list(
                zip(feature_names, shap_vals)
            )

            feature_contrib.sort(
                key=lambda x: abs(float(np.array(x[1]).flatten()[0])),
                reverse=True
            )

            explanations = []

            for name, val in feature_contrib[:5]:

                # Convert array -> float
                val = float(
                    np.array(val).flatten()[0]
                )

                direction = (
                    "increases"
                    if val > 0
                    else "decreases"
                )

                explanations.append(
                    f"{name} {direction} churn risk"
                )

            # =====================================
            # SAVE TO DATABASE
            # =====================================
            db = get_db()

            db.execute(
                """
                INSERT INTO predictions(
                    user_id,
                    customer_id,
                    input_data,
                    churn_result,
                    churn_probability
                )
                VALUES(?,?,?,?,?)
                """,
                (
                    session["user_id"],
                    customer_id,
                    json.dumps(form_data),
                    pred,
                    prob
                )
            )

            db.commit()

            # =====================================
            # FEATURE IMPORTANCE
            # =====================================
            fi_labels = list(
                feature_importances.keys()
            )

            fi_values = [
                round(v * 100, 2)
                for v in feature_importances.values()
            ]

            return render_template(
                "result.html",
                pred=pred,
                prob=round(prob * 100, 2),
                explanations=explanations,
                input_data=form_data,
                fi_labels=json.dumps(fi_labels),
                fi_values=json.dumps(fi_values),
                user=user
            )

        except Exception as e:

            print("PREDICTION ERROR:", e)

            raise e

    return render_template(
        "predict.html",
        user=user
    )




# =========================================================
# CSV UPLOAD
# =========================================================
@app.route("/upload-csv", methods=["GET", "POST"])
@login_required
def upload_csv():

    user = current_user()

    if request.method == "POST":

        try:

            file = request.files["file"]

            if not file:

                flash("No file uploaded", "danger")

                return redirect(
                    url_for("upload_csv")
                )

            # Read CSV
            df = pd.read_csv(file)

            results = []

            for _, row in df.iterrows():

                form_data = {}

                for feature in feature_names:

                    if feature in row:
                        form_data[feature] = row[feature]

                    else:
                        form_data[feature] = "No"

                input_array = preprocess_input(
                    form_data
                )

                input_scaled = scaler.transform(
                    input_array
                )

                pred = int(
                    model.predict(input_scaled)[0]
                )

                prob = float(
                    model.predict_proba(input_scaled)[0][1]
                )

                result = (
                    "CHURN"
                    if pred == 1
                    else "SAFE"
                )

                row_data = row.to_dict()

                row_data["Prediction"] = result

                row_data["Probability"] = round(
                    prob * 100,
                    2
                )

                results.append(row_data)

            result_df = pd.DataFrame(results)

            output_path = os.path.join(
                "instance",
                "predictions_output.csv"
            )

            result_df.to_csv(
                output_path,
                index=False
            )

            return render_template(
                "csv_result.html",
                tables=result_df.head(20).to_html(
                    classes="data-table",
                    index=False
                ),
                download_link=url_for(
                    "download_csv"
                ),
                user=user
            )

        except Exception as e:

            print("CSV ERROR:", e)

            raise e

    return render_template(
        "upload_csv.html",
        user=user
    )


# =========================================================
# DOWNLOAD CSV
# =========================================================
@app.route("/download-csv")
@login_required
def download_csv():

    path = os.path.join(
        "instance",
        "predictions_output.csv"
    )

    return send_file(
        path,
        as_attachment=True
    )


# =========================================================
# HISTORY
# =========================================================
@app.route("/history")
@login_required
def history():

    db = get_db()

    user = current_user()

    preds = db.execute(
        """
        SELECT *
        FROM predictions
        WHERE user_id=?
        ORDER BY id DESC
        """,
        (session["user_id"],)
    ).fetchall()

    return render_template(
        "history.html",
        predictions=preds,
        user=user
    )


# =========================================================
# ERROR PAGES
# =========================================================
@app.errorhandler(404)
def page_not_found(e):
    return render_template(
        "404.html"
    ), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template(
        "500.html"
    ), 500


# =========================================================
# RUN APP
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)