import React, { useState } from "react";
import axios from "axios";

function App() {
  const [grades, setGrades] = useState({});
  const [subjects, setSubjects] = useState(["MATEMATIK", "BAHASA MALAYSIA", "BAHASA INGGERIS", "SAINS", "SEJARAH"]);
  const [result, setResult] = useState("");

  const gradeOptions = ["A+", "A", "A-", "B+", "B", "B-", "C", "D", "E", "F"];

  // Handle grade changes for existing subjects
  const handleGradeChange = (subject, grade) => {
    setGrades({ ...grades, [subject]: grade });
  };

  // Add a new subject with an empty grade
  const handleAddSubject = () => {
    const newSubject = prompt("Enter the new subject name:");
    if (newSubject && !subjects.includes(newSubject)) {
      setSubjects([...subjects, newSubject]);
      setGrades({ ...grades, [newSubject]: "" });  // Set an empty grade initially
    } else {
      alert("Subject already exists or invalid name");
    }
  };

  // Remove a subject and its grade
  const handleRemoveSubject = (subject) => {
    setSubjects(subjects.filter((item) => item !== subject));
    const newGrades = { ...grades };
    delete newGrades[subject];
    setGrades(newGrades);
  };

  // Submit grades and fetch the recommended course(s)
  const handleSubmit = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:5000/recommend", grades, {
        headers: {
          "Content-Type": "application/json"
        }
      });

      // Log the response to check if the courses are returned correctly
      console.log("Recommended courses:", response.data.recommended_courses);

      // Display the recommended courses
      setResult(response.data.recommended_courses.join(", "));
    } catch (error) {
      console.error("Error fetching recommendations:", error);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Course Recommendation System</h1>

      {/* Display subjects and their corresponding grade input */}
      {subjects.map((subject) => (
        <div key={subject} style={{ marginBottom: "10px" }}>
          <label>{subject}: </label>
          <select
            value={grades[subject]}
            onChange={(e) => handleGradeChange(subject, e.target.value)}
          >
            <option value="">Select Grade</option>
            {gradeOptions.map((grade) => (
              <option key={grade} value={grade}>
                {grade}
              </option>
            ))}
          </select>
          <button onClick={() => handleRemoveSubject(subject)} style={{ marginLeft: "10px" }}>
            Remove
          </button>
        </div>
      ))}

      {/* Buttons for adding subjects and submitting the form */}
      <button onClick={handleAddSubject} style={{ marginBottom: "10px" }}>
        Add Subject
      </button>
      <br />
      <button onClick={handleSubmit}>Submit</button>

      {/* Display the result */}
      {result && <h2>Recommended Course(s): {result}</h2>}
    </div>
  );
}

export default App;
