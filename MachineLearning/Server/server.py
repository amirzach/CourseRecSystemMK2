from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

def grade_to_numeric(grade):
    grade_map = {'A+': 4.3, 'A': 4, 'A-': 3.7, 'B+': 3.3, 'B': 3, 'B-': 2.7, 'C': 2, 'D': 1, 'E': 0.5, 'F': 0}
    return grade_map.get(grade, 0)

def preprocess_data(file_path):
    try:
        # Load the data
        data = pd.read_excel(file_path)

        # Normalize column names
        data.columns = (
            data.columns.str.strip()  # Remove leading/trailing spaces
            .str.upper()              # Convert to uppercase
            .str.replace(r"\\n", " ", regex=True)  # Replace line breaks with space
            .str.replace(r"\\s+", " ", regex=True)  # Collapse multiple spaces
        )
        print("Normalized columns:", data.columns.tolist())  # Debugging step

        # Drop unnecessary columns
        columns_to_drop = ['ANGKA GILIRAN', 'NAMA']
        data_cleaned = data.drop(columns=columns_to_drop, errors='ignore')

        # Ensure all grades are numeric
        grade_columns = data_cleaned.columns.difference(['RECOMMENDED COURSES'])
        for col in grade_columns:
            data_cleaned[col] = pd.to_numeric(data_cleaned[col], errors='coerce').fillna(0)

        # Split 'RECOMMENDED_COURSES' into lists for multi-label classification
        data_cleaned['RECOMMENDED COURSES'] = data_cleaned['RECOMMENDED COURSES'].apply(
            lambda x: x.split(', ') if isinstance(x, str) else []
        )

        return data_cleaned
    except KeyError as e:
        raise KeyError(f"Column not found: {e}")
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
        if not recommended_courses:
            recommended_courses.append('General')
        return recommended_courses
    except Exception as e:
        raise Exception(f"Error in assign_course: {e}")

file_path = r'C:\Users\User\CourseRecMK3\Server\recommended_courses_ml_multiple.xlsx'
data_cleaned = preprocess_data(file_path)

X = data_cleaned.drop(columns=['RECOMMENDED COURSES'], errors='ignore')
y = pd.get_dummies(data_cleaned['RECOMMENDED COURSES'].apply(pd.Series).stack()).groupby(level=0).sum()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
course_labels = y.columns.tolist()

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        input_data = request.json

        # Convert grades to numeric
        for subject, grade in input_data.items():
            input_data[subject] = grade_to_numeric(grade)

        # Create a DataFrame from input data
        input_df = pd.DataFrame([input_data])

        # Ensure all features match the training data
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Default value for missing subjects

        # Reorder columns to match training data
        input_df = input_df[X.columns]

        # Make multi-label predictions
        predictions = model.predict(input_df)
        recommended_courses = [course_labels[i] for i, val in enumerate(predictions[0]) if val == 1]

        return jsonify({"recommended_courses": recommended_courses})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)