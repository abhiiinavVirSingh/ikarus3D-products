import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import RecommendationPage from './pages/RecommendationPage';
import AnalyticsPage from './pages/AnalyticsPage';
import './App.css'; // Optional: for basic layout styling

function App() {
  return (
    <Router>
      <div className="App">
        <nav>
          <ul>
            <li><Link to="/">Recommendations</Link></li>
            <li><Link to="/analytics">Analytics</Link></li>
          </ul>
        </nav>
        <Routes>
          <Route path="/" element={<RecommendationPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;