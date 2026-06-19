from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Regressors
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

app = Flask(__name__)


@app.route('/')
def serve_index():
    return render_template('index.html')


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

        df = pd.read_csv(file).dropna()
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # ==========================================
        # 🧠 THE "BRAIN": DETECT TASK TYPE
        # ==========================================
        is_numeric = pd.api.types.is_numeric_dtype(y)
        unique_count = y.nunique()

        if is_numeric and unique_count > 10:
            task_type = "regression"
        else:
            task_type = "classification"
            y = y.astype(str)  # Force categories to be text

        print(f"Detected Task: {task_type.upper()}")

        # Identify groupable columns (for fairness charts)
        groupable = [col for col in X.columns if X[col].nunique() < 10]
        if not groupable:
            groupable = X.columns.tolist()[:3]

        X_encoded = pd.get_dummies(X)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # ==========================================
        # ⚙️ LOAD THE RIGHT MODELS
        # ==========================================
        if task_type == "regression":
            models = {
                "Decision Tree (Simple)": DecisionTreeRegressor(max_depth=3, random_state=42),
                "Decision Tree (Complex)": DecisionTreeRegressor(max_depth=10, random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
                "Logistic Regression": LinearRegression()  # Kept the same name so UI colors match
            }
        else:
            models = {
                "Decision Tree (Simple)": DecisionTreeClassifier(max_depth=3, random_state=42),
                "Decision Tree (Complex)": DecisionTreeClassifier(max_depth=10, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=500)
            }

        accuracy_data = []
        importance_data = []
        predictions_df = X.loc[X_test.index].copy()
        predictions_df[target_col] = y_test

        # Train and Evaluate
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            predictions_df[name] = preds

            # ==========================================
            # 📊 RECORD THE RIGHT METRICS
            # ==========================================
            if task_type == "regression":
                mae = float(mean_absolute_error(y_test, preds))
                r2 = float(r2_score(y_test, preds))
                accuracy_data.append({"model": name, "mae": mae, "r2": r2})
            else:
                acc = float(accuracy_score(y_test, preds))
                accuracy_data.append({"model": name, "accuracy": acc})

            # Feature Importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, col in enumerate(X_encoded.columns):
                    if importances[i] > 0.01:
                        importance_data.append({
                            "model": name,
                            "feature": col,
                            "importance": float(importances[i])
                        })

        unique_classes = y.unique().tolist() if task_type == "classification" else []

        return jsonify({
            "success": True,
            "task_type": task_type,
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