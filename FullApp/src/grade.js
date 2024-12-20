import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const FirstApp = () => {
  const [grades, setGrades] = useState({
    MATEMATIK: '',
    'MATEMATIK TAMBAHAN': '',
    FIZIK: '',
    BIOLOGI: '',
    KIMIA: '',
    'PENDIDIKAN SENI VISUAL': '',
    EKONOMI: '',
    PERNIAGAAN: '',
    'PRINSIP PERAKAUNAN': '',
    'BAHASA INGGERIS': '',
    SEJARAH: '',
    'PENDIDIKAN ISLAM': '',
    'TASAWWUR ISLAM': '',
    'BAHASA MALAYSIA': '',
    MORAL: '',
  });

  const navigate = useNavigate();
  const [error, setError] = useState('');

  const gradeOptions = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C', 'D', 'E', 'F'];

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setGrades((prevGrades) => ({
      ...prevGrades,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/recommend', grades);
      navigate('/second', {
        state: {
          firstAppData: {
            recommendations: response.data.recommended_courses,
            accuracy: response.data.model_accuracy,
          },
          secondAppData: null,
        },
      });
    } catch (err) {
      setError(err.response ? err.response.data.error : 'An error occurred');
    }
  };

  return (
    <div style={{ 
      maxWidth: "600px", 
      margin: "50px auto", 
      padding: "20px", 
      borderRadius: "10px", 
      boxShadow: "0px 4px 8px rgba(0, 0, 0, 0.1)", 
      backgroundColor: "#f9f9f9", 
      fontFamily: "Arial, sans-serif" 
    }}>
      <h1 style={{ 
        textAlign: "center", 
        color: "#333", 
        marginBottom: "20px" 
      }}>
        SPM Grades-Based Recommendation
      </h1>
      <form onSubmit={handleSubmit}>
        {Object.keys(grades).map((subject) => (
          <div 
            key={subject} 
            style={{ marginBottom: "15px" }}
          >
            <label 
              style={{ 
                display: "block", 
                fontWeight: "bold", 
                marginBottom: "5px", 
                color: "#555" 
              }}>
              {subject}:
            </label>
            <select
              name={subject}
              value={grades[subject]}
              onChange={handleInputChange}
              style={{
                width: "100%",
                padding: "10px",
                borderRadius: "5px",
                border: "1px solid #ccc",
                fontSize: "16px",
                backgroundColor: "#fff",
              }}
            >
              <option value="">Select Grade</option>
              {gradeOptions.map((grade) => (
                <option key={grade} value={grade}>
                  {grade}
                </option>
              ))}
            </select>
          </div>
        ))}
        <button
          type="submit"
          style={{
            width: "100%",
            padding: "12px",
            backgroundColor: "#007BFF",
            color: "#fff",
            fontSize: "16px",
            fontWeight: "bold",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer",
            marginBottom: "15px",
          }}
        >
          Get Recommendations
        </button>
      </form>
      {error && (
        <p style={{ 
          color: "red", 
          textAlign: "center", 
          marginBottom: "15px" 
        }}>
          Error: {error}
        </p>
      )}
      <button
        onClick={() => navigate('/second')}
        style={{
          width: "100%",
          padding: "12px",
          backgroundColor: "#28A745",
          color: "#fff",
          fontSize: "16px",
          fontWeight: "bold",
          border: "none",
          borderRadius: "5px",
          cursor: "pointer",
        }}
      >
        Go to Interest-Based Recommendations
      </button>
    </div>
  );
};

export default FirstApp;
