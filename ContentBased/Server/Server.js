from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask_cors import CORS  # Added CORS import for handling CORS issues

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)  # This will allow requests from any origin

# Load data from a spreadsheet
def load_data(filepath):
    try:
        df = pd.read_excel(filepath)
        print("Columns in the dataset:", df.columns)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Train the Decision Tree Model
def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)  # Added random_state for reproducibility
    model.fit(X_train, y_train)
    return model

# Content-Based Filtering: Compute similarity scores
def content_based_filtering(df, user_answers):
    dataset_features = df.drop(columns=["Course"])  # Excluding 'Course' column from features
    similarity_scores = cosine_similarity([user_answers], dataset_features)
    most_similar_index = np.argmax(similarity_scores)
    return df.iloc[most_similar_index]["Course"]

@app.route('/recommend_course', methods=['POST'])
def recommend_course():
    data = request.get_json()
    user_answers = data['answers']
    
    # Load data from a spreadsheet
    filepath = r'C:\Users\User\CourseRecMK4\Server\QuestionnaireResultsHelper.xlsx'
    df = load_data(filepath)
    
    if df is None:
        return jsonify({"error": "Error loading data"}), 500  # Return an error if data could not be loaded
    
    # Prepare the dataset
    X = df.drop(columns=["Course"])  # Features (excluding the 'Course' column)
    y = df["Course"]  # Target (the 'Course' column)
    
    # Encode the target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split the dataset into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Train the Decision Tree
    model = train_decision_tree(X_train, y_train)
    
    # Convert user input to DataFrame with matching feature names
    user_answers_df = pd.DataFrame([user_answers], columns=X.columns)
    
    # Decision Tree Prediction
    prediction = model.predict(user_answers_df)
    recommended_course_dt = le.inverse_transform(prediction)[0]
    
    # Content-Based Filtering
    recommended_course_cb = content_based_filtering(df, user_answers)
    
    # Prepare the recommendation result
    result = {
        'decision_tree_recommendation': recommended_course_dt,
        'content_based_recommendation': recommended_course_cb
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
