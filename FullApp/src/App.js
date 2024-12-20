import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import FirstApp from "./grade";
import SecondApp from "./questionnaire";
import Recommendation from "./recommendation";

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<FirstApp />} />
        <Route path="/second" element={<SecondApp />} />
        <Route path="/recommendation" element={<Recommendation />} />
      </Routes>
    </Router>
  );
};

export default App;
