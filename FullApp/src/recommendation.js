import React from "react";
import { useLocation } from "react-router-dom";

const Recommendation = () => {
  const location = useLocation();
  const { firstAppData, secondAppData } = location.state || {};

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1>Combined Recommendations</h1>

      {firstAppData && (
        <div style={{ marginBottom: "20px" }}>
          <h2>SPM Grades-Based Recommendations:</h2>
          <ul>
            {firstAppData.recommendations.map((rec, index) => (
              <li key={index}>{rec}</li>
            ))}
          </ul>
          <p>Model Accuracy: {firstAppData.accuracy}</p>
        </div>
      )}

      {secondAppData && (
        <div>
          <h2>Interest-Based Recommendations:</h2>
          <p>
            <strong>Decision Tree:</strong> {secondAppData.decision_tree_recommendation}
          </p>
          <p>
            <strong>Content-Based Filtering:</strong> {secondAppData.content_based_recommendation}
          </p>
        </div>
      )}

      {!firstAppData && !secondAppData && (
        <p>No recommendations available. Please complete the previous steps.</p>
      )}
    </div>
  );
};

export default Recommendation;
