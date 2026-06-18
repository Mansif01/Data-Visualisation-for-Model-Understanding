# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Flask automatically knows to look in 'templates' for HTML and 'static' for JS!
app = Flask(__name__)

@app.route('/')
def serve_index():
    # Use render_template instead of send_from_directory
    return render_template('index.html')

# ==========================================
# Keep your /columns and /upload routes exactly the same below here!
# ==========================================
@app.route('/columns', methods=['POST'])
def get_columns():
    file = request.files['file']
    df = pd.read_csv(file)
    return jsonify({"columns": df.columns.tolist()})


@app.route('/upload', methods=['POST'])
def upload_and_train():
    try:
        file = request.files['file']
        target_col = request.form['target_col']

        # Load data and drop missing values
        df = pd.read_csv(file).dropna()

        # Separate Features (X) and Target (y)
        X = df.drop(target_col, axis=1)
        y = df[target_col].astype(str)  # Force target to be text for the dashboard

        # Identify groupable columns (categorical features with fewer than 10 unique values)
        groupable = [col for col in X.columns if X[col].nunique() < 10]
        if not groupable:
            groupable = X.columns.tolist()[:3]  # Fallback

        # One-Hot Encode text columns for the machine learning models
        X_encoded = pd.get_dummies(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # Define Models
        models = {
            "Decision Tree (Simple)": DecisionTreeClassifier(max_depth=3, random_state=42),
            "Decision Tree (Complex)": DecisionTreeClassifier(max_depth=10, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=500)
        }

        accuracy_data = []
        importance_data = []

        # Create the dashboard DataFrame (using original un-encoded text for readability)
        predictions_df = X.loc[X_test.index].copy()
        predictions_df[target_col] = y_test

        # Train and Predict
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Save predictions directly under the model's exact name!
            predictions_df[name] = preds

            # Record Accuracy
            acc = float(accuracy_score(y_test, preds))
            accuracy_data.append({"model": name, "accuracy": acc})

            # Record Feature Importance (skip Logistic Regression)
            if name != "Logistic Regression":
                importances = model.feature_importances_
                for i, col in enumerate(X_encoded.columns):
                    if importances[i] > 0.01:  # Only keep meaningful features to save space
                        importance_data.append({
                            "model": name,
                            "feature": col,
                            "importance": float(importances[i])
                        })

        # Get unique classes for the confusion matrix (e.g., [">50K", "<=50K"])
        unique_classes = y.unique().tolist()

        return jsonify({
            "success": True,
            "target_column": target_col,
            "groupable_columns": groupable,
            "classes": unique_classes,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": accuracy_data,
            "importance": importance_data,
            "predictions": predictions_df.to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == '__main__':
    print("🌟 Starting AutoML Server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)