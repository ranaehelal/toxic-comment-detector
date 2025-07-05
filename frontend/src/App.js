import React, { useState } from "react";
import axios from "axios";
import { Pie } from "react-chartjs-2";
import { motion } from "framer-motion";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
} from "chart.js";
ChartJS.register(ArcElement, Tooltip, Legend);

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const checkToxicity = async () => {
    setLoading(true);
    setResult(null);
    try {
      const response = await axios.post("http://localhost:8000/predict", {
        text,
        threshold: 0.5,
      });
      setResult(response.data);
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Something went wrong!");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setText("");
    setResult(null);
  };

  const chartData = result
    ? {
        labels: Object.keys(result.predictions),
        datasets: [
          {
            label: "Probability",
            data: Object.values(result.predictions),
            backgroundColor: [
              "#FF6384",
              "#36A2EB",
              "#FFCE56",
              "#4BC0C0",
              "#9966FF",
              "#FF9F40",
            ],
            borderWidth: 1,
          },
        ],
      }
    : null;

  return (
    <div className="container mt-5">
      <motion.h2
        className="text-center mb-4 title"
        initial={{ y: -30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
      >
        Toxic Comment Classifier
      </motion.h2>

      <div className="mb-3">
        <textarea
          className="form-control"
          rows="4"
          placeholder="Enter your comment here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
      </div>

      <div className="d-flex gap-3 mb-4">
        <button
          className="btn"
          style={{ backgroundColor: "#FF69B4", color: "white" }}
          onClick={checkToxicity}
          disabled={loading || text.trim() === ""}
        >
          {loading ? "Checking..." : "Check Toxicity"}
        </button>

        <button
          className="btn"
          style={{ backgroundColor: "#FF69B4", color: "white" }}
          onClick={handleReset}
          disabled={loading && !text}
        >
          Reset
        </button>
      </div>

      {result && (
        <motion.div
          className="card"
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
        >
          <div className="card-body">
            <h5>Result:</h5>
            <p>
              <strong>Is Toxic:</strong>{" "}
              <span className={result.is_toxic ? "text-danger" : "text-success"}>
                {result.is_toxic ? "Yes 😡" : "No 😊"}
              </span>
            </p>

            <h6 className="mt-3">Positive Labels:</h6>
            {result.positive_labels.length > 0 ? (
              <ul className="list-group mb-3">
                {result.positive_labels.map((label, index) => (
                  <li key={index} className="list-group-item d-flex justify-content-between">
                    <span>{label.label}</span>
                    <span className="badge bg-warning text-dark">
                      {(label.probability * 100).toFixed(2)}%
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-muted">No toxic labels detected.</p>
            )}

            <h6 className="mt-3">All Label Probabilities:</h6>
            {result.is_toxic && chartData ? (
              <div style={{ maxWidth: "400px", margin: "0 auto" }}>
                <Pie
                  data={chartData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                      legend: {
                        position: "bottom",
                        labels: {
                          font: { size: 14 },
                        },
                      },
                    },
                  }}
                />
              </div>
            ) : (
              <p className="text-muted">No toxic probabilities to display.</p>
            )}
          </div>
        </motion.div>
      )}
    </div>
  );
}

export default App;
