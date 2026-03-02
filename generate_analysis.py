import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

def generate_analysis():
    print("Loading dataset...")
    csv_path = os.path.join("data", "tripadvisor_hotel_reviews.csv")
    df = pd.read_csv(csv_path)

    # Ensure directories exist
    img_dir = os.path.join("static", "img")
    os.makedirs(img_dir, exist_ok=True)
    
    results_path = os.path.join("static", "model_results.json")
    stats_path = os.path.join("static", "analysis_stats.json")

    print("Generating Plots & Stats...")
    
    # === 1. Numerical Analysis & Boxplots ===
    numeric_df = df.select_dtypes(include=[np.number])
    stats_data = {}

    if not numeric_df.empty:
        # Calculate stats: Mean, Std, Min, Max
        stats_data["numerical_stats"] = numeric_df.describe().T[['mean', 'std', 'min', 'max']].round(2).to_dict(orient="index")
        
        # Boxplots for all numerical columns
        for col in numeric_df.columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=df[col], palette='Set2')
            plt.title(f'Boxplot of {col}')
            plt.savefig(os.path.join(img_dir, f"boxplot_{col}.png"))
            plt.close()
            
        # Feature Scaling Demonstration (StandardScaler)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)
        stats_data["scaled_stats"] = scaled_df.describe().T[['mean', 'std', 'min', 'max']].round(2).to_dict(orient="index")

    # === 2. Categorical Analysis ===
    cat_df = df.select_dtypes(include=['object'])
    if not cat_df.empty:
        stats_data["categorical_counts"] = {col: df[col].value_counts().to_dict() for col in cat_df.columns}

    # === 3. Visualizations ===
    # Univariate Analysis (Rating Distribution)
    if 'Rating' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(x='Rating', data=df, palette='viridis')
        plt.title('Distribution of Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.savefig(os.path.join(img_dir, "univariate_rating.png"))
        plt.close()

    # Bivariate Analysis (Rating vs Sentiment)
    if 'Rating' in df.columns:
        def get_sentiment(r):
            if r > 3: return 'Positive'
            elif r < 3: return 'Negative'
            else: return 'Neutral'
        df['Sentiment_Label'] = df['Rating'].apply(get_sentiment)
        
        # Save Target Distribution Plot
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Sentiment_Label', data=df, palette='pastel')
        plt.title('Distribution of Target Variable (Sentiment)')
        plt.savefig(os.path.join(img_dir, "target_dist.png"))
        plt.close()
        
        plt.figure(figsize=(8, 5))
        sns.countplot(x='Rating', hue='Sentiment_Label', data=df, palette='coolwarm')
        plt.title('Rating vs Sentiment')
        plt.savefig(os.path.join(img_dir, "bivariate_rating_sentiment.png"))
        plt.close()

    # Correlation Analysis (Heatmap)
    if not numeric_df.empty and numeric_df.shape[1] > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.savefig(os.path.join(img_dir, "correlation_heatmap.png"))
        plt.close()

    # Pairplot
    if not numeric_df.empty and numeric_df.shape[1] > 1:
        sns.pairplot(numeric_df)
        plt.savefig(os.path.join(img_dir, "pairplot.png"))
        plt.close()

    # Save Stats
    with open(stats_path, 'w') as f:
        json.dump(stats_data, f, indent=4)

    print("Training Models (Enhanced)...")
    model_results = []
    
    if 'Review' in df.columns and 'Rating' in df.columns:
        # Preprocessing
        def map_target(r):
            if r <= 2: return 0
            elif r == 3: return 1
            else: return 2
        
        y = df['Rating'].apply(map_target)
        X = df['Review'].fillna('')

        # Enhanced TF-IDF: N-grams, more features, sublinear tf
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2), 
            max_features=5000, 
            sublinear_tf=True, 
            stop_words='english'
        )
        X_tfidf = tfidf.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

        # Models with better parameters - Optimized for speed
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0)
        # SVC with probability=True is too slow. Using CalibratedClassifierCV with LinearSVC for probability if needed, 
        # or just LinearSVC for speed. For voting, we need probabilities or decision function.
        # Let's use SGDClassifier with log_loss for speed and probability.
        from sklearn.linear_model import SGDClassifier
        sgd = SGDClassifier(loss='log_loss', max_iter=1000, class_weight='balanced', random_state=42)
        
        nb = GaussianNB()
        dt = DecisionTreeClassifier(max_depth=20)
        
        # Ensemble Voting Classifier
        voting_clf = VotingClassifier(
            estimators=[('lr', lr), ('sgd', sgd)],
            voting='soft'
        )

        models = {
            "Logistic Regression (Tuned)": lr,
            "SGD Classifier (SVM-like)": sgd,
            "Naive Bayes": nb,
            "Decision Tree (Depth=20)": dt,
            "Voting Ensemble": voting_clf
        }

        for name, model in models.items():
            print(f"Training {name}...")
            try:
                start_time = time.time()
                if name == "Naive Bayes":
                    model.fit(X_train.toarray(), y_train)
                    y_pred = model.predict(X_test.toarray())
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                elapsed = time.time() - start_time

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                model_results.append({
                    "Model": name,
                    "Accuracy": round(acc, 4),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "F1 Score": round(f1, 4),
                    "Time (s)": round(elapsed, 2)
                })
            except Exception as e:
                print(f"Error training {name}: {e}")
                model_results.append({
                    "Model": name,
                    "Accuracy": "Error",
                    "Precision": "-", "Recall": "-", "F1 Score": "-", "Time (s)": "-"
                })

    # Save results
    with open(results_path, 'w') as f:
        json.dump(model_results, f, indent=4)
    
    print("Analysis Complete. Results saved.")

if __name__ == "__main__":
    generate_analysis()
