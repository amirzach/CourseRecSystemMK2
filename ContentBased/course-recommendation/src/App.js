import React, { useState } from "react";
import axios from "axios";

function App() {
  // State to store user answers
  const [answers, setAnswers] = useState(Array(25).fill(null)); // 25 questions
  const [recommendation, setRecommendation] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Questions to be displayed
  const questions = [
    {
      type: "multipleChoice",
      question: "Do you consider yourself more creative, analytical, or practical?",
      options: ["1-Creative", "2-Analytical", "3-Practical"],
    },
    {
      type: "multipleChoice",
      question: "Which set of skills or interests best describes you?",
      options: [
        "1. Problem-solving, logical thinking, and an interest in how things work.",
        "2. Curiosity about nature, scientific research, and exploring how the world works.",
        "3. Interest in biology, innovation, and working on solutions to health or environmental issues.",
        "4. Creativity in designing or making things, especially in food or other practical applications.",
        "5. Artistic talent, creativity, and a passion for visual expression.",
        "6. Business-minded, with an interest in economics, finance, or managing projects.",
        "7. Interest in technology, computers, and solving problems using logical approaches.",
        "8. Passion for history, law, or making a difference in society through governance or public service.",
        "9. Interest in teaching, religious studies, or exploring cultural traditions.",
        "10. Communication skills, creativity, and a passion for media, storytelling, or the arts.",
        "11. Interest in understanding human behavior, empathy, and helping others.",
        "12. Enjoy working with people, sharing knowledge, and guiding others.",
        "13. Love for exploring new places, cultures, and organizing travel experiences.",
      ],
    },
    {
      type: "multipleChoice",
      question: "How do you approach solving problems: step-by-step or intuitively?",
      options: ["1-Step-by-step", "2-Intuitively"],
    },
    {
      type: "multipleChoice",
      question: "Are you more comfortable working with data, people, or ideas?",
      options: ["1-Data", "2-People", "3-Ideas"],
    },
    {
      type: "multipleChoice",
      question: "How would you describe your learning style: visual, auditory, reading/writing, or kinesthetic?",
      options: ["1-Visual", "2-Auditory", "3-reading/writing","4-kinesthetic"],
    },    
    {
      type: "radioScale",
      question: "How confident are you in your mathematical skills?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Not Confident", "Very Confident"],
    },
    {
      type: "radioScale",
      question: "How would you rate your ability to understand and apply specific scientific concepts like biology, physics, and chemistry?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Poor", "Excellent"],
    },
    {
      type: "radioScale",
      question: "How proficient are you in solving additional mathematics problems?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Not Proficient", "Very Proficient"],
    },
    {
      type: "radioScale",
      question: "How would you rate your artistic or visual design abilities?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Very Weak", "Very Strong"],
    },
    {
      type: "radioScale",
      question: "How comfortable are you working on experiments or lab-based tasks?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Very Uncomfortable", "Very Comfortable"],
    },
    {
      type: "radioScale",
      question: "How would you rate your understanding of economics, accounting, or business concepts?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Very Weak", "Very Strong"],
    },
    {
      type: "radioScale",
      question: "How confident are you in your command of the English language, both written and spoken?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Not Confident", "Very Confident"],
    },
    {
      type: "radioScale",
      question: "How skilled are you at analyzing historical or legal concepts?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Not Skilled", "Highly Skilled"],
    },
    {
      type: "radioScale",
      question: "How strong is your grasp of Islamic Studies or moral concepts?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Very Weak", "Very Strong"],
    },
    {
      type: "radioScale",
      question: "How well do you communicate in Bahasa Malaysia?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Poorly", "Fluently"],
    },
    {
      type: "radioScale",
      question: "How much do you enjoy solving complex problems?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Not At All", "Very Much"],
    },
    {
      type: "radioScale",
      question: "How do you prefer hands-on, practical work or theoretical study?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Entirely theoretical", "Entirely Practical"],
    },
    {
      type: "radioScale",
      question: "How creative are you in coming up with new ideas or designs?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Not Creative", "Highly Creative"],
    },
    {
      type: "radioScale",
      question: "How comfortable are you working with technology and learning new software tools?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Not Comfortable", "Very Comfortable"],
    },
    {
      type: "radioScale",
      question: "Do you enjoy working with numbers and financial data?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Not At All", "Very Much"],
    },
    {
      type: "radioScale",
      question: "How much do you enjoy exploring human behavior or psychological concepts?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Not At All", "Very Much"],
    },
    {
      type: "radioScale",
      question: "Do you enjoy planning trips, learning about different cultures, or engaging in hospitality-related tasks?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Not At All", "Very Much"],
    },
    {
      type: "radioScale",
      question: "How confident are you in your ability to lead and manage projects or teams?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Not Confident", "Very Confident"],
    },
    {
      type: "radioScale",
      question: "Do you prefer working in structured environments or dynamic, creative spaces?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Entirely structured", "Entirely dynamic"],
    },
    {
      type: "radioScale",
      question: "How interested are you in contributing to society through teaching or educational programs?",
      options: ["1", "2", "3", "4", "5"],
      labels: ["Not At All", "Very Interested"],
    },
  ];
  
  const handleChange = (index, value) => {
    const newAnswers = [...answers];
    const numericalValue = value.split("-")[0];
    newAnswers[index] = parseInt(numericalValue, 10);
    setAnswers(newAnswers);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsSubmitting(true);

    if (answers.includes(null)) {
      alert("Please answer all the questions before submitting!");
      setIsSubmitting(false);
      return;
    }

    try {
      const response = await axios.post("http://localhost:5000/recommend_course", {
        answers,
      });

      setRecommendation(response.data);
    } catch (error) {
      console.error("There was an error submitting the form!", error);
      alert(`Error: ${error.response ? error.response.data.error : "Server unavailable"}`);
    }

    setIsSubmitting(false);
  };

  return (
    <div className="App">
      <h1>Course Recommendation Based on Your Interests</h1>
      <form onSubmit={handleSubmit}>
        {questions.map((question, index) => (
          <div key={index}>
            <label>{question.question}</label>
            <div>
              {question.options.map((option, i) => (
                <label key={i}>
                  <input
                    type="radio"
                    name={`question_${index}`}
                    value={option}
                    checked={answers[index] === parseInt(option.split("-")[0], 10)}
                    onChange={() => handleChange(index, option)}
                  />
                  {option}
                </label>
              ))}
            </div>
          </div>
        ))}
        <button type="submit" disabled={isSubmitting}>
          {isSubmitting ? "Submitting..." : "Submit"}
        </button>
      </form>

      {recommendation && (
        <div>
          <h2>Recommendation Results:</h2>
          <pre>{JSON.stringify(recommendation, null, 2)}</pre>
          <p>
            <strong>Decision Tree AI suggests:</strong> {recommendation.decision_tree_recommendation}
          </p>
          <p>
            <strong>Content-Based Filtering suggests:</strong> {recommendation.content_based_recommendation}
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
