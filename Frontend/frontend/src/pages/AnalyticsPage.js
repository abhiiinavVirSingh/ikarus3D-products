import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Bar, Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import './AnalyticsPage.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const API_URL = 'http://localhost:8000'; // Your FastAPI backend

function AnalyticsPage() {
  const [analyticsData, setAnalyticsData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAnalytics = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await axios.get(`${API_URL}/analytics`);
        setAnalyticsData(response.data);
      } catch (err) {
        console.error("Error fetching analytics:", err);
        setError(err.response?.data?.detail || "Failed to load analytics data.");
      } finally {
        setIsLoading(false);
      }
    };

    fetchAnalytics();
  }, []); // Empty dependency array means this runs once on mount

  // --- Chart Data Preparation ---
  const categoryChartData = {
    labels: analyticsData?.category_counts ? Object.keys(analyticsData.category_counts) : [],
    datasets: [
      {
        label: '# of Products',
        data: analyticsData?.category_counts ? Object.values(analyticsData.category_counts) : [],
        backgroundColor: [ // Add more colors if needed
          'rgba(255, 99, 132, 0.6)',
          'rgba(54, 162, 235, 0.6)',
          'rgba(255, 206, 86, 0.6)',
          'rgba(75, 192, 192, 0.6)',
          'rgba(153, 102, 255, 0.6)',
          'rgba(255, 159, 64, 0.6)',
        ],
        borderColor: [
           'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(153, 102, 255, 1)',
          'rgba(255, 159, 64, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false, // Allows setting height/width
     plugins: {
        legend: {
            position: 'top',
        },
        title: {
            display: true,
            text: 'Products per Category',
        },
    },
  };


  return (
    <div className="analytics-page">
      <h1>Product Analytics</h1>
      {isLoading && <p>Loading analytics...</p>}
      {error && <p className="error-message">Error: {error}</p>}
      {analyticsData && (
        <div className="analytics-content">
          <div className="chart-container">
             <h2>Product Counts by Top-Level Category</h2>
             <Bar data={categoryChartData} options={chartOptions}/>
             {/* You could use a Pie chart as well: <Pie data={categoryChartData} options={chartOptions} /> */}
          </div>

          <div className="stats-container">
            <h2>Price Statistics</h2>
            {analyticsData.price_distribution ? (
              <ul>
                <li><strong>Min Price:</strong> ${analyticsData.price_distribution.min?.toFixed(2) ?? 'N/A'}</li>
                <li><strong>Max Price:</strong> ${analyticsData.price_distribution.max?.toFixed(2) ?? 'N/A'}</li>
                <li><strong>Average Price:</strong> ${analyticsData.price_distribution.mean?.toFixed(2) ?? 'N/A'}</li>
                <li><strong>Median Price:</strong> ${analyticsData.price_distribution.median?.toFixed(2) ?? 'N/A'}</li>
              </ul>
            ) : <p>Price data not available.</p>}
          </div>

          {/* Add more charts/visualizations here */}

        </div>
      )}
    </div>
  );
}

export default AnalyticsPage;