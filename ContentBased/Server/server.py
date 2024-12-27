from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

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
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=5,  # Prevent overfitting
        min_samples_split=10,
        min_samples_leaf=5
    )
    model.fit(X_train, y_train)
    return model

# Evaluate Decision Tree using Cross-Validation
def evaluate_with_cross_validation(X, y):
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5
    )
    scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
    print("Cross-Validation Scores:", scores)
    print("Mean Accuracy:", scores.mean())
    return scores.mean()

# Content-Based Filtering: Compute similarity scores
def content_based_filtering(df, user_answers):
    # Normalize the dataset features
    dataset_features = df.drop(columns=["Course"])
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(dataset_features)
    user_answers_scaled = scaler.transform([user_answers])
    
    # Compute cosine similarity scores
    similarity_scores = cosine_similarity(user_answers_scaled, normalized_features)
    print("Similarity Scores:", similarity_scores)
    
    # Find indices with the highest similarity score
    most_similar_indices = np.argwhere(similarity_scores == np.amax(similarity_scores)).flatten()
    print("Most Similar Indices:", most_similar_indices)
    
    # Break ties randomly if there are multiple max indices
    most_similar_index = np.random.choice(most_similar_indices)
    
    # Return the course name corresponding to the most similar features
    if similarity_scores.max() < 0.5:  # Introduce a fallback threshold
        recommended_course = np.random.choice(df["Course"].unique())
        print("Fallback Recommendation triggered due to low similarity score.")
    else:
        recommended_course = df.iloc[most_similar_index]["Course"]
    return recommended_course

@app.route('/recommend_course', methods=['POST'])
def recommend_course():
    data = request.get_json()
    user_answers = data['answers']
    
    # Load data from a spreadsheet
    filepath = r'C:\Users\User\CourseRecMK4\Server\QuestionnaireResultsHelper.xlsx'
    df = load_data(filepath)
    
    if df is None:
        return jsonify({"error": "Error loading data"}), 500
    
    # Prepare the dataset
    X = df.drop(columns=["Course"])
    y = df["Course"]
    
    # Encode the target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Evaluate the model using cross-validation (optional, for debugging)
    cross_val_accuracy = evaluate_with_cross_validation(X, y_encoded)
    
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
    
    # Calculate accuracy of the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100  # Accuracy in percentage
    
    # Save new user answers and the predicted course to the spreadsheet
    new_row = pd.DataFrame([user_answers + [recommended_course_cb]], columns=X.columns.tolist() + ["Course"])
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Save the updated dataframe back to the Excel file
    try:
        df.to_excel(filepath, index=False)
        print("Data saved successfully")
    except Exception as e:
        print(f"Error saving data: {e}")

    # Prepare the recommendation result
    result = {
        'decision_tree_recommendation': recommended_course_dt,
        'content_based_recommendation': recommended_course_cb,
        'accuracy': accuracy,
        'cross_val_accuracy': cross_val_accuracy
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
