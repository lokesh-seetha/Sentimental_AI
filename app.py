from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this' # Needed for session management

# === DATABASE CONFIG ===
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///hotel_reviews.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# === LOAD TRAINED ML MODEL ===
MODEL_PATH = os.path.join("models", "sentiment_pipeline.pkl")
sentiment_model = joblib.load(MODEL_PATH)


# === DATABASE MODELS ===
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    reviews = db.relationship('Review', backref='author', lazy=True)

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(20), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # Nullable for backward compatibility or anonymous reviews if needed

    # star ratings (1–5)
    rating_overall = db.Column(db.Integer, nullable=True)
    rating_cleanliness = db.Column(db.Integer, nullable=True)
    rating_food = db.Column(db.Integer, nullable=True)
    rating_staff = db.Column(db.Integer, nullable=True)
    rating_amenities = db.Column(db.Integer, nullable=True)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()
    # Create Admin User if not exists
    admin_email = "admin@gmail.com"
    if not User.query.filter_by(email=admin_email).first():
        admin_user = User(email=admin_email, password=generate_password_hash("admin", method='scrypt'))
        db.session.add(admin_user)
        db.session.commit()
        print("Admin user created.")


# simple rule-based override for obvious negative phrases
NEGATIVE_PHRASES = [
    "not good", "not comfortable", "no comfort", "worst",
    "rooms are bad", "rooms are not good", "bad food",
    "food negative", "terrible", "horrible", "very bad"
]


def postprocess_sentiment(text: str, model_label: str) -> str:
    """If the review clearly contains negative patterns, force negative."""
    text_low = text.lower()
    if any(p in text_low for p in NEGATIVE_PHRASES):
        return "negative"
    return model_label


# === ROUTES ===

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid email or password')
    return render_template('login.html')

@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists')
            return redirect(url_for('register'))
            
        new_user = User(email=email, password=generate_password_hash(password, method='scrypt'))
        db.session.add(new_user)
        db.session.commit()
        
        login_user(new_user)
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route("/my_reviews")
@login_required
def my_reviews():
    if current_user.email == "admin@gmail.com":
        reviews = Review.query.all()
    else:
        reviews = Review.query.filter_by(user_id=current_user.id).all()
    return render_template("my_reviews.html", reviews=reviews)

@app.route("/add_review", methods=["POST"])
@login_required
def add_review():
    review_text = request.form.get("review", "").strip()

    # ratings from form (convert to int, handle empty)
    def get_int(name):
        value = request.form.get(name)
        return int(value) if value and value.isdigit() else None

    r_overall = get_int("rating_overall")
    r_clean = get_int("rating_cleanliness")
    r_food = get_int("rating_food")
    r_staff = get_int("rating_staff")
    r_amen = get_int("rating_amenities")

    if review_text:
        # model prediction
        model_label = sentiment_model.predict([review_text])[0]
        sentiment = postprocess_sentiment(review_text, model_label)

        new_review = Review(
            text=review_text,
            sentiment=sentiment,
            rating_overall=r_overall,
            rating_cleanliness=r_clean,
            rating_food=r_food,
            rating_staff=r_staff,
            rating_amenities=r_amen,
            user_id=current_user.id
        )
        db.session.add(new_review)
        db.session.commit()

    return redirect(url_for("my_reviews"))


@app.route("/analytics")
@login_required
def analytics():
    is_admin = (current_user.email == "admin@gmail.com")
    
    if is_admin:
        reviews = Review.query.all()
    else:
        reviews = Review.query.filter_by(user_id=current_user.id).all()

    # sentiment distribution
    # Filter locally since we already fetched 'reviews' based on permissions
    sentiment_counts = {
        "positive": sum(1 for r in reviews if r.sentiment == "positive"),
        "negative": sum(1 for r in reviews if r.sentiment == "negative"),
        "neutral": sum(1 for r in reviews if r.sentiment == "neutral")
    }

    n = len(reviews)

    # default averages
    avg_ratings = {
        "overall": 0.0,
        "cleanliness": 0.0,
        "food": 0.0,
        "staff": 0.0,
        "amenities": 0.0,
    }

    if n > 0:
        sum_overall = sum(r.rating_overall or 0 for r in reviews)
        sum_clean = sum(r.rating_cleanliness or 0 for r in reviews)
        sum_food = sum(r.rating_food or 0 for r in reviews)
        sum_staff = sum(r.rating_staff or 0 for r in reviews)
        sum_amen = sum(r.rating_amenities or 0 for r in reviews)

        # avoid divide by zero by counting only non-null ratings
        cnt_overall = sum(1 for r in reviews if r.rating_overall)
        cnt_clean = sum(1 for r in reviews if r.rating_cleanliness)
        cnt_food = sum(1 for r in reviews if r.rating_food)
        cnt_staff = sum(1 for r in reviews if r.rating_staff)
        cnt_amen = sum(1 for r in reviews if r.rating_amenities)

        if cnt_overall:
            avg_ratings["overall"] = round(sum_overall / cnt_overall, 2)
        if cnt_clean:
            avg_ratings["cleanliness"] = round(sum_clean / cnt_clean, 2)
        if cnt_food:
            avg_ratings["food"] = round(sum_food / cnt_food, 2)
        if cnt_staff:
            avg_ratings["staff"] = round(sum_staff / cnt_staff, 2)
        if cnt_amen:
            avg_ratings["amenities"] = round(sum_amen / cnt_amen, 2)

    # complaint counts by aspect (rating <= 2 considered complaint)
    complaint_counts = {
        "cleanliness": 0,
        "food": 0,
        "staff": 0,
        "amenities": 0,
    }
    for r in reviews:
        if r.rating_cleanliness and r.rating_cleanliness <= 2:
            complaint_counts["cleanliness"] += 1
        if r.rating_food and r.rating_food <= 2:
            complaint_counts["food"] += 1
        if r.rating_staff and r.rating_staff <= 2:
            complaint_counts["staff"] += 1
        if r.rating_amenities and r.rating_amenities <= 2:
            complaint_counts["amenities"] += 1

    return render_template(
        "analytics.html",
        sentiment_counts=sentiment_counts,
        avg_ratings=avg_ratings,
        complaint_counts=complaint_counts,
        is_admin=is_admin,
        reviews=reviews
    )


@app.route("/api/sentiment_data")
def sentiment_data():
    data = {
        "positive": Review.query.filter_by(sentiment="positive").count(),
        "negative": Review.query.filter_by(sentiment="negative").count(),
        "neutral": Review.query.filter_by(sentiment="neutral").count()
    }
    return jsonify(data)


@app.route("/output")
@login_required
def output_page():
    if current_user.email != "admin@gmail.com":
        flash("Access denied. Admin only.")
        return redirect(url_for('index'))
    # 1. List of imported libraries (key ones)
    libraries = ["Flask", "SQLAlchemy", "pandas", "numpy", "textblob", "joblib", "os"]

    # 2. Read Dataset
    csv_path = os.path.join("data", "tripadvisor_hotel_reviews.csv")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error loading dataset: {e}"

    # 3. Data Analysis
    # Shape
    dataset_shape = df.shape

    # Head (first 5 rows) -> convert to HTML or list of dicts
    dataset_head = df.head().to_dict(orient="records")
    columns = df.columns.tolist()

    # Missing values
    missing_values = df.isnull().sum().to_dict()

    # Duplicates
    duplicates_count = df.duplicated().sum()

    # Data Types
    data_types = df.dtypes.astype(str).to_dict()

    # Descriptive Statistics
    desc_stats = df.describe().to_html(classes="table table-striped table-bordered", float_format="%.2f")

    # Outliers in 'Rating'
    outliers_info = "Column 'Rating' not found."
    if 'Rating' in df.columns and pd.api.types.is_numeric_dtype(df['Rating']):
        Q1 = df['Rating'].quantile(0.25)
        Q3 = df['Rating'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df['Rating'] < lower_bound) | (df['Rating'] > upper_bound)]
        outliers_count = outliers.shape[0]
        outliers_info = {
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Count": outliers_count
        }

    # === EDA: Generate Plots ===
    # Plots are generated offline by generate_analysis.py
    # We just check if they exist and pass the paths
    plots = {
        'univariate': "img/univariate_rating.png",
        'bivariate': "img/bivariate_rating_sentiment.png",
        'correlation': "img/correlation_heatmap.png",
        'pairplot': "img/pairplot.png",
        'target_dist': "img/target_dist.png"
    }
    
    # Boxplots
    boxplots = []
    if os.path.exists(os.path.join("static", "img")):
        for f in os.listdir(os.path.join("static", "img")):
            if f.startswith("boxplot_"):
                boxplots.append(f"img/{f}")

    # Load Analysis Stats
    analysis_stats = {}
    stats_path = os.path.join("static", "analysis_stats.json")
    if os.path.exists(stats_path):
        import json
        with open(stats_path, 'r') as f:
            analysis_stats = json.load(f)
            # Remove categorical counts as requested
            if "categorical_counts" in analysis_stats:
                del analysis_stats["categorical_counts"]

    # === MODEL BUILDING ===
    # Load pre-computed results
    model_results = []
    results_path = os.path.join("static", "model_results.json")
    if os.path.exists(results_path):
        import json
        with open(results_path, 'r') as f:
            model_results = json.load(f)
    else:
        model_results = [{"Model": "Error", "Accuracy": "Run generate_analysis.py first"}]

    return render_template(
        "output.html",
        libraries=libraries,
        dataset_shape=dataset_shape,
        dataset_head=dataset_head,
        columns=columns,
        missing_values=missing_values,
        duplicates_count=duplicates_count,
        data_types=data_types,
        outliers_info=outliers_info,
        desc_stats=desc_stats,
        plots=plots,
        boxplots=boxplots,
        analysis_stats=analysis_stats,
        model_results=model_results
    )


if __name__ == "__main__":
    app.run(debug=True)
