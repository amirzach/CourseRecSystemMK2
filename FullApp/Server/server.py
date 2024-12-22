from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from openpyxl import load_workbook
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

########################################
# Functionality from First server.py   #
########################################

def grade_to_numeric(grade):
    grade_map = {'A+': 4.3, 'A': 4, 'A-': 3.7, 'B+': 3.3, 'B': 3, 'B-': 2.7, 'C': 2, 'D': 1, 'E': 0.5, 'F': 0}
    return grade_map.get(grade, 0)

def preprocess_data(file_path):
    try:
        data = pd.read_excel(file_path)
        data.columns = (
            data.columns.str.strip()
            .str.upper()
            .str.replace(r"\\n", " ", regex=True)
            .str.replace(r"\\s+", " ", regex=True)
        )
        grade_columns = data.columns.difference(['RECOMMENDED COURSES'])
        for col in grade_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        data['RECOMMENDED COURSES'] = data['RECOMMENDED COURSES'].apply(
            lambda x: x.split(', ') if isinstance(x, str) else []
        )
        return data
    except Exception as e:
        raise Exception(f"Error in preprocess_data: {e}")

def assign_course(row):
    try:
        recommended_courses = []
        if row.get('MATEMATIK', 0) >= 3.7 and row.get('MATEMATIK TAMBAHAN', 0) >= 3.7 and row.get('FIZIK', 0) >= 3:
            recommended_courses.append('Engineering')
        if row.get('BIOLOGI', 0) >= 3.7 and row.get('FIZIK', 0) >= 3 and row.get('KIMIA', 0) >= 3:
            recommended_courses.append('Science')
        if row.get('BIOLOGI', 0) >= 3.5 and row.get('KIMIA', 0) >= 3.5:
            recommended_courses.append('Biotechnology')
        if row.get('BIOLOGI', 0) >= 3 and row.get('KIMIA', 0) >= 3 and row.get('PENDIDIKAN SENI VISUAL', 0) >= 3:
            recommended_courses.append('Food Technology')
        if row.get('PENDIDIKAN SENI VISUAL', 0) >= 3.7 and row.get('BAHASA INGGERIS', 0) >= 3:
            recommended_courses.append('Fine Arts and Design')
        if row.get('EKONOMI', 0) >= 3.5 or row.get('PERNIAGAAN', 0) >= 3.5 or row.get('PRINSIP PERAKAUNAN', 0) >= 3.5:
            recommended_courses.append('Commerce')
        if row.get('MATEMATIK', 0) >= 3.5 and row.get('BAHASA INGGERIS', 0) >= 3:
            recommended_courses.append('Information Technology')
        if row.get('SEJARAH', 0) >= 3 and row.get('BAHASA INGGERIS', 0) >= 3:
            recommended_courses.append('Law and Policing')
        if row.get('PENDIDIKAN ISLAM', 0) >= 3.5 or row.get('TASAWWUR ISLAM', 0) >= 3.5:
            recommended_courses.append('Islamic Studies and TESL')
        if row.get('BAHASA MALAYSIA', 0) >= 3 and row.get('BAHASA INGGERIS', 0) >= 3 and row.get('PENDIDIKAN SENI VISUAL', 0) >= 3:
            recommended_courses.append('Arts and Media')
        if row.get('BIOLOGI', 0) >= 3 and row.get('MORAL', 0) >= 3:
            recommended_courses.append('Psychology and Health')
        if row.get('BAHASA INGGERIS', 0) >= 3.5 and row.get('PENDIDIKAN ISLAM', 0) >= 3.5:
            recommended_courses.append('Education')
        if row.get('PERNIAGAAN', 0) >= 3.5 and row.get('SEJARAH', 0) >= 3.5:
            recommended_courses.append('Travel and Hospitality')
        return recommended_courses
    except Exception as e:
        raise Exception(f"Error in assign_course: {e}")

file_path1 = r'C:\Users\User\CourseRecMK5\Server\recommended_courses_ml_multiple.xlsx'
data_cleaned = preprocess_data(file_path1)

X = data_cleaned.drop(columns=['RECOMMENDED COURSES'], errors='ignore')
y = pd.get_dummies(data_cleaned['RECOMMENDED COURSES'].apply(pd.Series).stack()).groupby(level=0).sum()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
course_labels = y.columns.tolist()

def calculate_accuracy():
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

accuracy_percentage = calculate_accuracy() * 100

def append_to_spreadsheet(input_data, recommended_courses):
    try:
        workbook = load_workbook(file_path1)
        sheet = workbook.active
        row_data = [input_data.get(subject, "") for subject in X.columns]
        row_data.append(", ".join(recommended_courses))
        sheet.append(row_data)
        workbook.save(file_path1)
    except Exception as e:
        raise Exception(f"Error updating spreadsheet: {e}")

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        input_data = request.json
        for subject, grade in input_data.items():
            input_data[subject] = grade_to_numeric(grade)
        input_df = pd.DataFrame([input_data])
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X.columns]
        predictions = model.predict(input_df)
        predicted_courses = [course_labels[i] for i in range(len(course_labels)) if predictions[0][i] == 1]
        recommended_courses = list(set(predicted_courses))
        append_to_spreadsheet(input_data, recommended_courses)
        return jsonify({
            "recommended_courses": recommended_courses,
            "model_accuracy": f"{accuracy_percentage:.2f}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

########################################
# Functionality from Second server.py  #
########################################

def load_data(filepath):
    try:
        df = pd.read_excel(filepath)
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def content_based_filtering(df, user_answers):
    dataset_features = df.drop(columns=["Course"])
    similarity_scores = cosine_similarity([user_answers], dataset_features)
    most_similar_index = np.argmax(similarity_scores)
    return df.iloc[most_similar_index]["Course"]

@app.route('/recommend_course', methods=['POST'])
def recommend_course():
    try:
        data = request.get_json()
        user_answers = data['answers']
        filepath2 = r'C:\Users\User\CourseRecMK5\Server\QuestionnaireResultsHelper.xlsx'
        df = load_data(filepath2)
        X = df.drop(columns=["Course"])
        y = df["Course"]
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        model = train_decision_tree(X_train, y_train)
        user_answers_df = pd.DataFrame([user_answers], columns=X.columns)
        prediction = model.predict(user_answers_df)
        recommended_course_dt = le.inverse_transform(prediction)[0]
        recommended_course_cb = content_based_filtering(df, user_answers)
        result = {
            'decision_tree_recommendation': recommended_course_dt,
            'content_based_recommendation': recommended_course_cb
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/final_recommendation', methods=['POST'])
def final_recommendation():
    try:
        # Input data from request
        input_data = request.get_json()
        
        # Fetch first app data recommendations
        for subject, grade in input_data.get("grades", {}).items():
            input_data["grades"][subject] = grade_to_numeric(grade)
        input_df = pd.DataFrame([input_data["grades"]])
        input_df = input_df[X.columns]
        predictions = model.predict(input_df)
        predicted_courses = [course_labels[i] for i in range(len(course_labels)) if predictions[0][i] == 1]
        
        # Fetch second app data recommendations
        user_answers = input_data.get("answers", [])
        df = load_data(filepath2)
        X = df.drop(columns=["Course"])
        y = df["Course"]
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        model_dt = train_decision_tree(X_train, y_train)
        user_answers_df = pd.DataFrame([user_answers], columns=X.columns)
        prediction = model_dt.predict(user_answers_df)
        recommended_course_dt = le.inverse_transform(prediction)[0]
        recommended_course_cb = content_based_filtering(df, user_answers)
        
        # Match courses between both recommendations
        matched_courses = set(predicted_courses) & {recommended_course_dt, recommended_course_cb}
        final_courses = list(matched_courses) if matched_courses else [recommended_course_dt, recommended_course_cb]
        
        return jsonify({
            "first_app_data": {
                "recommendations": predicted_courses,
                "model_accuracy": f"{accuracy_percentage:.2f}%"
            },
            "second_app_data": {
                "decision_tree_recommendation": recommended_course_dt,
                "content_based_recommendation": recommended_course_cb
            },
            "final_recommendations": final_courses
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500   

if __name__ == "__main__":
    app.run(debug=True)
