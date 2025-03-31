import React, { useState, useEffect } from "react";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

function App() {
  const [images, setImages] = useState([]);
  const [processedImages, setProcessedImages] = useState(null);
  const [tableData, setTableData] = useState([]);
  const [bestAlgo, setBestAlgo] = useState("");
  const [metricExplanations, setMetricExplanations] = useState({});
  const [algorithmExplanations, setAlgorithmExplanations] = useState({});
  const [selectionReason, setSelectionReason] = useState("");
  const [error, setError] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const imagesPerPage = 20;

  useEffect(() => {
    const loadImages = async () => {
      try {
        const response = await fetch("http://localhost:5000/images");
        if (!response.ok) throw new Error("Backend is down!");
        const data = await response.json();
        setImages(data.images);
      } catch (err) {
        setError(err.message);
        toast.error(`Error: ${err.message}`);
      }
    };
    loadImages();
  }, []);

  const handleDragStart = (e, imgSrc) => {
    e.dataTransfer.setData("imageSrc", imgSrc);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    const imgSrc = e.dataTransfer.getData("imageSrc");
    if (!imgSrc) {
      toast.error("No image selected!");
      return;
    }

    const formData = new FormData();
    try {
      const response = await fetch(imgSrc);
      const blob = await response.blob();
      const imageName = imgSrc.split("/").pop();
      formData.append("image", blob, imageName);

      toast.info("Detecting...");
      const result = await fetch("http://localhost:5000/detect", {
        method: "POST",
        body: formData,
      });
      if (!result.ok)
        throw new Error((await result.json()).error || "Detection failed!");
      const data = await result.json();
      setProcessedImages(data.images);
      setTableData(data.table);
      setBestAlgo(data.best_algo);
      setMetricExplanations(data.metric_explanations);
      setAlgorithmExplanations(data.algorithm_explanations);
      setSelectionReason(data.selection_reason);
      setError(null);
      toast.success("Detection complete!");
    } catch (err) {
      setError(err.message);
      toast.error(`Error: ${err.message}`);
    }
  };

  const handleDragOver = (e) => e.preventDefault();

  const indexOfLastImage = currentPage * imagesPerPage;
  const indexOfFirstImage = indexOfLastImage - imagesPerPage;
  const currentImages = images.slice(indexOfFirstImage, indexOfLastImage);
  const totalPages = Math.ceil(images.length / imagesPerPage);

  const nextPage = () =>
    currentPage < totalPages && setCurrentPage(currentPage + 1);
  const prevPage = () => currentPage > 1 && setCurrentPage(currentPage - 1);

  return (
    <div style={{ textAlign: "center", fontFamily: "Arial, sans-serif" }}>
      <h1 style={{ fontSize: "2.5em", margin: "20px 0", color: "#333" }}>
        Drone Detection Research
      </h1>
      {error && (
        <p style={{ color: "red", fontSize: "1.2em" }}>Error: {error}</p>
      )}
      <ToastContainer />
      <div
        style={{ display: "flex", flexWrap: "wrap", justifyContent: "center" }}
      >
        {currentImages.map((img, index) => (
          <img
            key={index}
            src={img}
            alt={`Drone ${indexOfFirstImage + index}`}
            draggable
            onDragStart={(e) => handleDragStart(e, img)}
            style={{
              width: "100px",
              height: "100px",
              margin: "10px",
              objectFit: "cover",
            }}
          />
        ))}
      </div>
      <div style={{ margin: "20px 0" }}>
        <button
          onClick={prevPage}
          disabled={currentPage === 1}
          style={buttonStyle}
        >
          Previous
        </button>
        <span style={{ margin: "0 10px" }}>
          Page {currentPage} of {totalPages}
        </span>
        <button
          onClick={nextPage}
          disabled={currentPage === totalPages}
          style={buttonStyle}
        >
          Next
        </button>
      </div>
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        style={{
          width: "400px",
          border: "2px dashed #666",
          margin: "20px auto",
          padding: "20px",
          backgroundColor: "#f9f9f9",
        }}
      >
        {processedImages ? (
          <div>
            {["rcnn", "yolov5", "yolov7", "yolov9"].map((algo) => (
              <div key={algo} style={{ marginBottom: "20px" }}>
                <h3 style={{ fontSize: "1.2em", color: "#444" }}>
                  {algo.toUpperCase()}
                </h3>
                <img
                  src={processedImages[algo]}
                  alt={algo}
                  style={{ width: "300px" }}
                />
              </div>
            ))}
          </div>
        ) : (
          <p style={{ fontSize: "1.2em", color: "#666" }}>Drag Image Here</p>
        )}
      </div>
      {tableData.length > 0 && (
        <div style={{ margin: "20px 0" }}>
          <h2 style={{ fontSize: "1.8em", color: "#333" }}>Comparison Table</h2>
          <table
            style={{
              borderCollapse: "collapse",
              margin: "20px auto",
              width: "80%",
            }}
          >
            <thead>
              <tr style={{ backgroundColor: "#e0e0e0" }}>
                {Object.keys(tableData[0]).map((key) => (
                  <th key={key} style={tableHeaderStyle}>
                    {key}
                    <br />
                    <small style={{ fontSize: "0.8em", color: "#555" }}>
                      {metricExplanations[key] || ""}
                    </small>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableData.map((row, index) => (
                <tr
                  key={index}
                  style={
                    row.Algorithm === bestAlgo
                      ? { backgroundColor: "#d4edda" }
                      : {}
                  }
                >
                  {Object.values(row).map((value, i) => (
                    <td key={i} style={tableCellStyle}>
                      {typeof value === "number" ? value.toFixed(2) : value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <h3 style={{ fontSize: "1.5em", color: "#28a745" }}>
            Best Algorithm: {bestAlgo}
          </h3>
          <div style={{ margin: "20px auto", width: "80%", textAlign: "left" }}>
            <h4 style={{ fontSize: "1.3em", color: "#444" }}>
              Algorithm Comparisons
            </h4>
            {Object.entries(algorithmExplanations).map(
              ([algo, explanation]) => (
                <div key={algo} style={{ marginBottom: "20px" }}>
                  <h5
                    style={{
                      fontSize: "1.1em",
                      color: algo === bestAlgo ? "#28a745" : "#555",
                    }}
                  >
                    {algo}
                  </h5>
                  <pre
                    style={{
                      whiteSpace: "pre-wrap",
                      fontSize: "1em",
                      color: "#555",
                    }}
                  >
                    {explanation}
                  </pre>
                </div>
              )
            )}
          </div>
          <div style={{ margin: "20px auto", width: "80%", textAlign: "left" }}>
            <h4 style={{ fontSize: "1.3em", color: "#444" }}>
              Why These Algorithms?
            </h4>
            <pre
              style={{ whiteSpace: "pre-wrap", fontSize: "1em", color: "#555" }}
            >
              {selectionReason}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

const buttonStyle = {
  padding: "10px 20px",
  margin: "0 10px",
  cursor: "pointer",
  backgroundColor: "#007bff",
  color: "white",
  border: "none",
  borderRadius: "5px",
};

const tableHeaderStyle = {
  border: "1px solid #ccc",
  padding: "10px",
  fontWeight: "bold",
  backgroundColor: "#f2f2f2",
};

const tableCellStyle = {
  border: "1px solid #ccc",
  padding: "10px",
};

export default App;
