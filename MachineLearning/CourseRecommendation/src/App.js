import React, { useState } from 'react';
import axios from 'axios';

const App = () => {
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

    const [recommendations, setRecommendations] = useState([]);
    const [error, setError] = useState('');

    // Update grades based on user input
    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setGrades((prevGrades) => ({
            ...prevGrades,
            [name]: value,
        }));
    };

    // Submit grades and get recommendations
    const handleSubmit = async (e) => {
        e.preventDefault();

        try {
            const response = await axios.post('http://localhost:5000/recommend', grades);

            // Ensure recommendations are an array
            if (response.data.recommended_courses && Array.isArray(response.data.recommended_courses)) {
                setRecommendations(response.data.recommended_courses);
                setError('');
            } else {
                throw new Error('Unexpected response format');
            }
        } catch (err) {
            setError(err.response ? err.response.data.error : 'An error occurred');
            setRecommendations([]);
        }
    };

    return (
        <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
            <h1>SPM Course Recommendation System</h1>
            <form onSubmit={handleSubmit} style={{ marginBottom: '20px' }}>
                {Object.keys(grades).map((subject) => (
                    <div key={subject} style={{ marginBottom: '10px' }}>
                        <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '5px' }}>
                            {subject}:
                        </label>
                        <input
                            type="text"
                            name={subject}
                            value={grades[subject]}
                            onChange={handleInputChange}
                            placeholder="Enter grade (e.g., A+, A, B+)"
                            style={{
                                width: '100%',
                                padding: '8px',
                                borderRadius: '4px',
                                border: '1px solid #ccc',
                            }}
                        />
                    </div>
                ))}
                <button
                    type="submit"
                    style={{
                        padding: '10px 20px',
                        backgroundColor: '#007BFF',
                        color: '#fff',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                    }}
                >
                    Get Recommendations
                </button>
            </form>

            {error && (
                <div style={{ color: 'red', marginBottom: '20px' }}>
                    <strong>Error:</strong> {error}
                </div>
            )}

            {recommendations.length > 0 && (
                <div>
                    <h2>Recommended Courses:</h2>
                    <ul>
                        {recommendations.map((course, index) => (
                            <li key={index} style={{ marginBottom: '10px' }}>
                                {course}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

export default App;
