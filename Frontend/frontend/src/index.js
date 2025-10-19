import React from 'react';
import ReactDOM from 'react-dom/client'; // Use this import for React 18+
import './index.css'; // Imports global CSS styles
import App from './App'; // Imports the main application component
import reportWebVitals from './reportWebVitals'; // Optional: for performance measurement

// Find the root DOM node provided in the public/index.html file
const rootElement = document.getElementById('root');
const root = ReactDOM.createRoot(rootElement);

// Render the main App component inside React's Strict Mode for checks
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to measure performance, you can pass a function
// like console.log or send data to an analytics service.
// Learn more: https://bit.ly/CRA-vitals
reportWebVitals();