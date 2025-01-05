from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def load_data(filepath):
    try:
        df = pd.read_excel(filepath)
        print("Columns in the dataset:", df.columns)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=8,           # Increased depth for better accuracy
        min_samples_split=5,   # Reduced to allow more splits
        min_samples_leaf=2,    # Reduced to allow smaller leaf nodes
        criterion='entropy'     # Using entropy instead of gini for better splits
    )
    model.fit(X_train, y_train)
    return model

def evaluate_with_cross_validation(X, y):
    dt_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        criterion='entropy'
    )
    scores = cross_val_score(dt_model, X, y, cv=5)
    print("Cross-Validation Scores:", scores)
    print("Mean Accuracy:", scores.mean())
    return scores.mean()

@app.route('/recommend_course', methods=['POST'])
def recommend_course():
    data = request.get_json()
    user_answers = data['answers']
    
    filepath = r'C:\Users\User\CourseRecMK4\Server\QuestionnaireResultsHelper.xlsx'
    df = load_data(filepath)
    
    if df is None:
        return jsonify({"error": "Error loading data"}), 500
    
    X = df.drop(columns=["Course"])
    y = df["Course"]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    cross_val_accuracy = evaluate_with_cross_validation(X_scaled, y_encoded)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    # Train Decision Tree model
    dt_model = train_decision_tree(X_train, y_train)
    
    # Get probability scores for all classes
    user_answers_scaled = scaler.transform([user_answers])
    dt_probabilities = dt_model.predict_proba(user_answers_scaled)[0]
    
    # Get top 3 recommendations based on decision tree probabilities
    top_indices = np.argsort(dt_probabilities)[-3:][::-1]
    decision_tree_recommendations = [
        {
            'course': le.inverse_transform([idx])[0],
            'probability': float(dt_probabilities[idx])
        }
        for idx in top_indices
    ]
    
    # Calculate model accuracy
    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    
    # Save new user answers with the top recommendation
    new_row = pd.DataFrame([user_answers + [decision_tree_recommendations[0]['course']]], 
                          columns=X.columns.tolist() + ["Course"])
    df = pd.concat([df, new_row], ignore_index=True)
    
    try:
        df.to_excel(filepath, index=False)
        print("Data saved successfully")
    except Exception as e:
        print(f"Error saving data: {e}")

    result = {
        'decision_tree_recommendations': decision_tree_recommendations,
        'accuracy': accuracy,
        'cross_val_accuracy': cross_val_accuracy
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
