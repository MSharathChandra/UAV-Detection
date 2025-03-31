// import React, { useState, useEffect } from "react";
// import { ToastContainer, toast } from "react-toastify";
// import "react-toastify/dist/ReactToastify.css";

// function App() {
//   const [images, setImages] = useState([]);
//   const [processedImages, setProcessedImages] = useState(null);
//   const [tableData, setTableData] = useState([]);
//   const [bestAlgo, setBestAlgo] = useState("");
//   const [metricExplanations, setMetricExplanations] = useState({});
//   const [algorithmExplanations, setAlgorithmExplanations] = useState({});
//   const [selectionReason, setSelectionReason] = useState("");
//   const [error, setError] = useState(null);
//   const [currentPage, setCurrentPage] = useState(1);
//   const imagesPerPage = 20;

//   useEffect(() => {
//     const loadImages = async () => {
//       try {
//         const response = await fetch("http://localhost:5000/images");
//         if (!response.ok) throw new Error("Backend is down!");
//         const data = await response.json();
//         setImages(data.images);
//       } catch (err) {
//         setError(err.message);
//         toast.error(`Error: ${err.message}`);
//       }
//     };
//     loadImages();
//   }, []);

//   const handleDragStart = (e, imgSrc) => {
//     e.dataTransfer.setData("imageSrc", imgSrc);
//   };

//   const handleDrop = async (e) => {
//     e.preventDefault();
//     const imgSrc = e.dataTransfer.getData("imageSrc");
//     if (!imgSrc) {
//       toast.error("No image selected!");
//       return;
//     }

//     const formData = new FormData();
//     try {
//       const response = await fetch(imgSrc);
//       const blob = await response.blob();
//       const imageName = imgSrc.split("/").pop();
//       formData.append("image", blob, imageName);

//       toast.info("Detecting...");
//       const result = await fetch("http://localhost:5000/detect", {
//         method: "POST",
//         body: formData,
//       });
//       if (!result.ok)
//         throw new Error((await result.json()).error || "Detection failed!");
//       const data = await result.json();
//       setProcessedImages(data.images);
//       setTableData(data.table);
//       setBestAlgo(data.best_algo);
//       setMetricExplanations(data.metric_explanations);
//       setAlgorithmExplanations(data.algorithm_explanations);
//       setSelectionReason(data.selection_reason);
//       setError(null);
//       toast.success("Detection complete!");
//     } catch (err) {
//       setError(err.message);
//       toast.error(`Error: ${err.message}`);
//     }
//   };

//   const handleDragOver = (e) => e.preventDefault();

//   const indexOfLastImage = currentPage * imagesPerPage;
//   const indexOfFirstImage = indexOfLastImage - imagesPerPage;
//   const currentImages = images.slice(indexOfFirstImage, indexOfLastImage);
//   const totalPages = Math.ceil(images.length / imagesPerPage);

//   const nextPage = () =>
//     currentPage < totalPages && setCurrentPage(currentPage + 1);
//   const prevPage = () => currentPage > 1 && setCurrentPage(currentPage - 1);

//   return (
//     <div style={{ textAlign: "center", fontFamily: "Arial, sans-serif" }}>
//       <h1 style={{ fontSize: "2.5em", margin: "20px 0", color: "#333" }}>
//         Drone Detection Research
//       </h1>
//       {error && (
//         <p style={{ color: "red", fontSize: "1.2em" }}>Error: {error}</p>
//       )}
//       <ToastContainer />
//       <div
//         style={{ display: "flex", flexWrap: "wrap", justifyContent: "center" }}
//       >
//         {currentImages.map((img, index) => (
//           <img
//             key={index}
//             src={img}
//             alt={`Drone ${indexOfFirstImage + index}`}
//             draggable
//             onDragStart={(e) => handleDragStart(e, img)}
//             style={{
//               width: "100px",
//               height: "100px",
//               margin: "10px",
//               objectFit: "cover",
//             }}
//           />
//         ))}
//       </div>
//       <div style={{ margin: "20px 0" }}>
//         <button
//           onClick={prevPage}
//           disabled={currentPage === 1}
//           style={buttonStyle}
//         >
//           Previous
//         </button>
//         <span style={{ margin: "0 10px" }}>
//           Page {currentPage} of {totalPages}
//         </span>
//         <button
//           onClick={nextPage}
//           disabled={currentPage === totalPages}
//           style={buttonStyle}
//         >
//           Next
//         </button>
//       </div>
//       <div
//         onDrop={handleDrop}
//         onDragOver={handleDragOver}
//         style={{
//           width: "400px",
//           border: "2px dashed #666",
//           margin: "20px auto",
//           padding: "20px",
//           backgroundColor: "#f9f9f9",
//         }}
//       >
//         {processedImages ? (
//           <div>
//             {["rcnn", "yolov5", "yolov7", "yolov9"].map((algo) => (
//               <div key={algo} style={{ marginBottom: "20px" }}>
//                 <h3 style={{ fontSize: "1.2em", color: "#444" }}>
//                   {algo.toUpperCase()}
//                 </h3>
//                 <img
//                   src={processedImages[algo]}
//                   alt={algo}
//                   style={{ width: "300px" }}
//                 />
//               </div>
//             ))}
//           </div>
//         ) : (
//           <p style={{ fontSize: "1.2em", color: "#666" }}>Drag Image Here</p>
//         )}
//       </div>
//       {tableData.length > 0 && (
//         <div style={{ margin: "20px 0" }}>
//           <h2 style={{ fontSize: "1.8em", color: "#333" }}>Comparison Table</h2>
//           <table
//             style={{
//               borderCollapse: "collapse",
//               margin: "20px auto",
//               width: "80%",
//             }}
//           >
//             <thead>
//               <tr style={{ backgroundColor: "#e0e0e0" }}>
//                 {Object.keys(tableData[0]).map((key) => (
//                   <th key={key} style={tableHeaderStyle}>
//                     {key}
//                     <br />
//                     <small style={{ fontSize: "0.8em", color: "#555" }}>
//                       {metricExplanations[key] || ""}
//                     </small>
//                   </th>
//                 ))}
//               </tr>
//             </thead>
//             <tbody>
//               {tableData.map((row, index) => (
//                 <tr
//                   key={index}
//                   style={
//                     row.Algorithm === bestAlgo
//                       ? { backgroundColor: "#d4edda" }
//                       : {}
//                   }
//                 >
//                   {Object.values(row).map((value, i) => (
//                     <td key={i} style={tableCellStyle}>
//                       {typeof value === "number" ? value.toFixed(2) : value}
//                     </td>
//                   ))}
//                 </tr>
//               ))}
//             </tbody>
//           </table>
//           <h3 style={{ fontSize: "1.5em", color: "#28a745" }}>
//             Best Algorithm: {bestAlgo}
//           </h3>
//           <div style={{ margin: "20px auto", width: "80%", textAlign: "left" }}>
//             <h4 style={{ fontSize: "1.3em", color: "#444" }}>
//               Algorithm Comparisons
//             </h4>
//             {Object.entries(algorithmExplanations).map(
//               ([algo, explanation]) => (
//                 <div key={algo} style={{ marginBottom: "20px" }}>
//                   <h5
//                     style={{
//                       fontSize: "1.1em",
//                       color: algo === bestAlgo ? "#28a745" : "#555",
//                     }}
//                   >
//                     {algo}
//                   </h5>
//                   <pre
//                     style={{
//                       whiteSpace: "pre-wrap",
//                       fontSize: "1em",
//                       color: "#555",
//                     }}
//                   >
//                     {explanation}
//                   </pre>
//                 </div>
//               )
//             )}
//           </div>
//           <div style={{ margin: "20px auto", width: "80%", textAlign: "left" }}>
//             <h4 style={{ fontSize: "1.3em", color: "#444" }}>
//               Why These Algorithms?
//             </h4>
//             <pre
//               style={{ whiteSpace: "pre-wrap", fontSize: "1em", color: "#555" }}
//             >
//               {selectionReason}
//             </pre>
//           </div>
//         </div>
//       )}
//     </div>
//   );
// }

// const buttonStyle = {
//   padding: "10px 20px",
//   margin: "0 10px",
//   cursor: "pointer",
//   backgroundColor: "#007bff",
//   color: "white",
//   border: "none",
//   borderRadius: "5px",
// };

// const tableHeaderStyle = {
//   border: "1px solid #ccc",
//   padding: "10px",
//   fontWeight: "bold",
//   backgroundColor: "#f2f2f2",
// };

// const tableCellStyle = {
//   border: "1px solid #ccc",
//   padding: "10px",
// };

// export default App;

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
  const [labeledImages, setLabeledImages] = useState(null);
  const [numImagesProcessed, setNumImagesProcessed] = useState(0);
  const [rangeInput, setRangeInput] = useState("");
  const [customImages, setCustomImages] = useState([]);
  const [activeTab, setActiveTab] = useState("single");
  const imagesPerPage = 20;

  useEffect(() => {
    const loadImages = async () => {
      try {
        const response = await fetch("http://localhost:5000/images");
        if (!response.ok) throw new Error("Backend is down ra!");
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
      toast.error("No image selected ra!");
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
        throw new Error((await result.json()).error || "Detection failed ra!");
      const data = await result.json();
      setProcessedImages(data.images);
      setTableData(data.table);
      setBestAlgo(data.best_algo);
      setMetricExplanations(data.metric_explanations);
      setAlgorithmExplanations(data.algorithm_explanations);
      setSelectionReason(data.selection_reason);
      setLabeledImages(null);
      setNumImagesProcessed(0);
      setError(null);
      toast.success("Detection completed!");
    } catch (err) {
      setError(err.message);
      toast.error(`Error: ${err.message}`);
    }
  };

  const handleDetectAll = async () => {
    try {
      toast.info("Detecting all images...");
      const result = await fetch("http://localhost:5000/detect_all", {
        method: "GET",
      });
      if (!result.ok)
        throw new Error((await result.json()).error || "Detection failed ra!");
      const data = await result.json();
      setProcessedImages(null);
      setTableData(data.table);
      setBestAlgo(data.best_algo);
      setMetricExplanations(data.metric_explanations);
      setAlgorithmExplanations(data.algorithm_explanations);
      setSelectionReason(data.selection_reason);
      setLabeledImages(data.labeled_images);
      setNumImagesProcessed(data.num_images_processed);
      setError(null);
      toast.success("All images detected ra!");
    } catch (err) {
      setError(err.message);
      toast.error(`Error: ${err.message}`);
    }
  };

  const handleDetectRange = async () => {
    if (!rangeInput || isNaN(rangeInput) || rangeInput <= 0) {
      toast.error("Enter a valid number ra!");
      return;
    }

    try {
      toast.info(`Detecting ${rangeInput} images...`);
      const result = await fetch("http://localhost:5000/detect_range", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ num_images: parseInt(rangeInput) }),
      });
      if (!result.ok)
        throw new Error((await result.json()).error || "Detection failed ra!");
      const data = await result.json();
      setProcessedImages(null);
      setTableData(data.table);
      setBestAlgo(data.best_algo);
      setMetricExplanations(data.metric_explanations);
      setAlgorithmExplanations(data.algorithm_explanations);
      setSelectionReason(data.selection_reason);
      setLabeledImages(data.labeled_images);
      setNumImagesProcessed(data.num_images_processed);
      setError(null);
      toast.success(`${rangeInput} images detected ra!`);
    } catch (err) {
      setError(err.message);
      toast.error(`Error: ${err.message}`);
    }
  };

  const handleCustomDrop = (e) => {
    e.preventDefault();
    const imgSrc = e.dataTransfer.getData("imageSrc");
    if (!imgSrc) {
      toast.error("No image dropped ra!");
      return;
    }
    const imageName = imgSrc.split("/").pop();
    if (!customImages.includes(imgSrc)) {
      setCustomImages((prev) => [...prev, imgSrc]);
      toast.success(`Added ${imageName} to custom list ra!`);
    } else {
      toast.info(`${imageName} already added ra!`);
    }
  };

  const handleDetectCustom = async () => {
    if (!customImages.length) {
      toast.error("No custom images selected ra!");
      return;
    }

    const imageNames = customImages.map((url) => url.split("/").pop());
    try {
      toast.info("Detecting custom images...");
      const result = await fetch("http://localhost:5000/detect_custom", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ images: imageNames }),
      });
      if (!result.ok)
        throw new Error((await result.json()).error || "Detection failed ra!");
      const data = await result.json();
      setProcessedImages(null);
      setTableData(data.table);
      setBestAlgo(data.best_algo);
      setMetricExplanations(data.metric_explanations);
      setAlgorithmExplanations(data.algorithm_explanations);
      setSelectionReason(data.selection_reason);
      setLabeledImages(data.labeled_images);
      setNumImagesProcessed(data.num_images_processed);
      setError(null);
      toast.success("Custom images detected ra!");
    } catch (err) {
      setError(err.message);
      toast.error(`Error: ${err.message}`);
    }
  };

  const handleClear = () => {
    setProcessedImages(null);
    setTableData([]);
    setBestAlgo("");
    setMetricExplanations({});
    setAlgorithmExplanations({});
    setSelectionReason("");
    setLabeledImages(null);
    setNumImagesProcessed(0);
    setRangeInput("");
    setCustomImages([]);
    setError(null);
    toast.success("Cleared! Ready for new detection.");
  };

  const handleDragOver = (e) => e.preventDefault();

  const indexOfLastImage = currentPage * imagesPerPage;
  const indexOfFirstImage = indexOfLastImage - imagesPerPage;
  const currentImages = images.slice(indexOfFirstImage, indexOfLastImage);
  const totalPages = Math.ceil(images.length / imagesPerPage);

  const nextPage = () =>
    currentPage < totalPages && setCurrentPage(currentPage + 1);
  const prevPage = () => currentPage > 1 && setCurrentPage(currentPage - 1);

  const formatValue = (key, value) => {
    if (
      ["Precision", "Recall", "mAP"].includes(key) &&
      typeof value === "number"
    ) {
      return `${(value <= 1 ? value * 100 : value).toFixed(2)}%`;
    }
    return typeof value === "number" ? value.toFixed(2) : value;
  };

  return (
    <div
      style={{
        maxWidth: "1200px",
        margin: "0 auto",
        padding: "20px",
        fontFamily: "Arial, sans-serif",
        backgroundColor: "#f4f4f4",
      }}
    >
      <div
        style={{
          textAlign: "center",
          padding: "20px 0",
          backgroundColor: "#007bff",
          color: "white",
          borderRadius: "8px 8px 0 0",
        }}
      >
        <h1 style={{ margin: "0", fontSize: "2.5em" }}>
          Drone Detection Research
        </h1>
        {error && (
          <p style={{ color: "#ff4d4d", fontSize: "1.2em", margin: "10px 0" }}>
            Error: {error}
          </p>
        )}
        <ToastContainer />
      </div>

      <div
        style={{
          display: "flex",
          justifyContent: "center",
          backgroundColor: "#fff",
          borderBottom: "2px solid #ddd",
          marginBottom: "20px",
        }}
      >
        <button
          style={{
            padding: "15px 30px",
            cursor: "pointer",
            backgroundColor: activeTab === "single" ? "#007bff" : "#f0f0f0",
            color: activeTab === "single" ? "white" : "black",
            border: "none",
            fontSize: "1.1em",
            transition: "background-color 0.3s",
          }}
          onClick={() => setActiveTab("single")}
          onMouseOver={(e) =>
            (e.target.style.backgroundColor =
              activeTab === "single" ? "#007bff" : "#e0e0e0")
          }
          onMouseOut={(e) =>
            (e.target.style.backgroundColor =
              activeTab === "single" ? "#007bff" : "#f0f0f0")
          }
        >
          Single Image
        </button>
        <button
          style={{
            padding: "15px 30px",
            cursor: "pointer",
            backgroundColor: activeTab === "all" ? "#007bff" : "#f0f0f0",
            color: activeTab === "all" ? "white" : "black",
            border: "none",
            fontSize: "1.1em",
            transition: "background-color 0.3s",
          }}
          onClick={() => setActiveTab("all")}
          onMouseOver={(e) =>
            (e.target.style.backgroundColor =
              activeTab === "all" ? "#007bff" : "#e0e0e0")
          }
          onMouseOut={(e) =>
            (e.target.style.backgroundColor =
              activeTab === "all" ? "#007bff" : "#f0f0f0")
          }
        >
          Detect All
        </button>
        <button
          style={{
            padding: "15px 30px",
            cursor: "pointer",
            backgroundColor: activeTab === "range" ? "#007bff" : "#f0f0f0",
            color: activeTab === "range" ? "white" : "black",
            border: "none",
            fontSize: "1.1em",
            transition: "background-color 0.3s",
          }}
          onClick={() => setActiveTab("range")}
          onMouseOver={(e) =>
            (e.target.style.backgroundColor =
              activeTab === "range" ? "#007bff" : "#e0e0e0")
          }
          onMouseOut={(e) =>
            (e.target.style.backgroundColor =
              activeTab === "range" ? "#007bff" : "#f0f0f0")
          }
        >
          Detect Range
        </button>
        <button
          style={{
            padding: "15px 30px",
            cursor: "pointer",
            backgroundColor: activeTab === "custom" ? "#007bff" : "#f0f0f0",
            color: activeTab === "custom" ? "white" : "black",
            border: "none",
            fontSize: "1.1em",
            transition: "background-color 0.3s",
          }}
          onClick={() => setActiveTab("custom")}
          onMouseOver={(e) =>
            (e.target.style.backgroundColor =
              activeTab === "custom" ? "#007bff" : "#e0e0e0")
          }
          onMouseOut={(e) =>
            (e.target.style.backgroundColor =
              activeTab === "custom" ? "#007bff" : "#f0f0f0")
          }
        >
          Custom Images
        </button>
      </div>

      <div
        style={{
          backgroundColor: "#fff",
          padding: "20px",
          borderRadius: "8px",
          boxShadow: "0 2px 5px rgba(0, 0, 0, 0.1)",
        }}
      >
        {activeTab === "single" && (
          <div>
            <h2
              style={{ fontSize: "1.8em", color: "#333", marginBottom: "20px" }}
            >
              Drag Single Image
            </h2>
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: "10px",
                justifyContent: "center",
              }}
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
                    objectFit: "cover",
                    borderRadius: "5px",
                    cursor: "move",
                  }}
                />
              ))}
            </div>
            <div style={{ margin: "20px 0", textAlign: "center" }}>
              <button
                onClick={prevPage}
                disabled={currentPage === 1}
                style={{
                  padding: "8px 16px",
                  margin: "0 5px",
                  backgroundColor: currentPage === 1 ? "#ccc" : "#007bff",
                  color: "white",
                  border: "none",
                  borderRadius: "5px",
                  cursor: currentPage === 1 ? "not-allowed" : "pointer",
                }}
              >
                Previous
              </button>
              <span style={{ margin: "0 10px" }}>
                Page {currentPage} of {totalPages}
              </span>
              <button
                onClick={nextPage}
                disabled={currentPage === totalPages}
                style={{
                  padding: "8px 16px",
                  margin: "0 5px",
                  backgroundColor:
                    currentPage === totalPages ? "#ccc" : "#007bff",
                  color: "white",
                  border: "none",
                  borderRadius: "5px",
                  cursor:
                    currentPage === totalPages ? "not-allowed" : "pointer",
                }}
              >
                Next
              </button>
            </div>
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              style={{
                border: "2px dashed #666",
                padding: "20px",
                textAlign: "center",
                backgroundColor: "#f9f9f9",
                margin: "20px 0",
                minHeight: "200px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {processedImages ? (
                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: "20px",
                    justifyContent: "center",
                  }}
                >
                  {["rcnn", "yolov5", "yolov7", "yolov9"].map((algo) => (
                    <div key={algo} style={{ textAlign: "center" }}>
                      <h3
                        style={{
                          fontSize: "1.2em",
                          color: "#444",
                          marginBottom: "10px",
                        }}
                      >
                        {algo.toUpperCase()}
                      </h3>
                      <img
                        src={processedImages[algo]}
                        alt={algo}
                        style={{ width: "300px", borderRadius: "5px" }}
                      />
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ fontSize: "1.2em", color: "#666" }}>
                  Drag Image Here
                </p>
              )}
            </div>
            <button
              onClick={handleClear}
              style={{
                padding: "12px 24px",
                backgroundColor: "#dc3545",
                color: "white",
                border: "none",
                borderRadius: "5px",
                cursor: "pointer",
                fontSize: "1.1em",
                transition: "background-color 0.3s",
                marginTop: "20px",
              }}
              onMouseOver={(e) => (e.target.style.backgroundColor = "#c82333")}
              onMouseOut={(e) => (e.target.style.backgroundColor = "#dc3545")}
            >
              Clear
            </button>
          </div>
        )}

        {activeTab === "all" && (
          <div>
            <h2
              style={{ fontSize: "1.8em", color: "#333", marginBottom: "20px" }}
            >
              Detect All Images
            </h2>
            <button
              onClick={handleDetectAll}
              style={{
                padding: "12px 24px",
                backgroundColor: "#28a745",
                color: "white",
                border: "none",
                borderRadius: "5px",
                cursor: "pointer",
                fontSize: "1.1em",
                transition: "background-color 0.3s",
              }}
              onMouseOver={(e) => (e.target.style.backgroundColor = "#218838")}
              onMouseOut={(e) => (e.target.style.backgroundColor = "#28a745")}
            >
              Detect All
            </button>
            <button
              onClick={handleClear}
              style={{
                padding: "12px 24px",
                backgroundColor: "#dc3545",
                color: "white",
                border: "none",
                borderRadius: "5px",
                cursor: "pointer",
                fontSize: "1.1em",
                transition: "background-color 0.3s",
                marginLeft: "20px",
              }}
              onMouseOver={(e) => (e.target.style.backgroundColor = "#c82333")}
              onMouseOut={(e) => (e.target.style.backgroundColor = "#dc3545")}
            >
              Clear
            </button>
          </div>
        )}

        {activeTab === "range" && (
          <div>
            <h2
              style={{ fontSize: "1.8em", color: "#333", marginBottom: "20px" }}
            >
              Detect Range
            </h2>
            <div
              style={{ display: "flex", justifyContent: "center", gap: "10px" }}
            >
              <input
                type="number"
                value={rangeInput}
                onChange={(e) => setRangeInput(e.target.value)}
                placeholder="Enter number of images"
                style={{
                  padding: "8px",
                  fontSize: "1em",
                  border: "1px solid #ddd",
                  borderRadius: "5px",
                }}
              />
              <button
                onClick={handleDetectRange}
                style={{
                  padding: "12px 24px",
                  backgroundColor: "#28a745",
                  color: "white",
                  border: "none",
                  borderRadius: "5px",
                  cursor: "pointer",
                  fontSize: "1.1em",
                  transition: "background-color 0.3s",
                }}
                onMouseOver={(e) =>
                  (e.target.style.backgroundColor = "#218838")
                }
                onMouseOut={(e) => (e.target.style.backgroundColor = "#28a745")}
              >
                Detect Range
              </button>
            </div>
            <button
              onClick={handleClear}
              style={{
                padding: "12px 24px",
                backgroundColor: "#dc3545",
                color: "white",
                border: "none",
                borderRadius: "5px",
                cursor: "pointer",
                fontSize: "1.1em",
                transition: "background-color 0.3s",
                marginTop: "20px",
              }}
              onMouseOver={(e) => (e.target.style.backgroundColor = "#c82333")}
              onMouseOut={(e) => (e.target.style.backgroundColor = "#dc3545")}
            >
              Clear
            </button>
          </div>
        )}

        {activeTab === "custom" && (
          <div>
            <h2
              style={{ fontSize: "1.8em", color: "#333", marginBottom: "20px" }}
            >
              Custom Images
            </h2>
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: "10px",
                justifyContent: "center",
              }}
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
                    objectFit: "cover",
                    borderRadius: "5px",
                    cursor: "move",
                  }}
                />
              ))}
            </div>
            <div style={{ margin: "20px 0", textAlign: "center" }}>
              <button
                onClick={prevPage}
                disabled={currentPage === 1}
                style={{
                  padding: "8px 16px",
                  margin: "0 5px",
                  backgroundColor: currentPage === 1 ? "#ccc" : "#007bff",
                  color: "white",
                  border: "none",
                  borderRadius: "5px",
                  cursor: currentPage === 1 ? "not-allowed" : "pointer",
                }}
              >
                Previous
              </button>
              <span style={{ margin: "0 10px" }}>
                Page {currentPage} of {totalPages}
              </span>
              <button
                onClick={nextPage}
                disabled={currentPage === totalPages}
                style={{
                  padding: "8px 16px",
                  margin: "0 5px",
                  backgroundColor:
                    currentPage === totalPages ? "#ccc" : "#007bff",
                  color: "white",
                  border: "none",
                  borderRadius: "5px",
                  cursor:
                    currentPage === totalPages ? "not-allowed" : "pointer",
                }}
              >
                Next
              </button>
            </div>
            <div
              onDrop={handleCustomDrop}
              onDragOver={handleDragOver}
              style={{
                border: "2px dashed #666",
                padding: "20px",
                textAlign: "center",
                backgroundColor: "#f9f9f9",
                margin: "20px 0",
                minHeight: "200px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {customImages.length > 0 ? (
                <div style={{ display: "flex", flexWrap: "wrap", gap: "10px" }}>
                  {customImages.map((img, index) => (
                    <img
                      key={index}
                      src={img}
                      alt={`Custom ${index}`}
                      style={{
                        width: "80px",
                        height: "80px",
                        objectFit: "cover",
                        borderRadius: "5px",
                      }}
                    />
                  ))}
                </div>
              ) : (
                <p style={{ fontSize: "1.2em", color: "#666" }}>
                  Drag Dataset Images Here
                </p>
              )}
            </div>
            <button
              onClick={handleDetectCustom}
              style={{
                padding: "12px 24px",
                backgroundColor: "#28a745",
                color: "white",
                border: "none",
                borderRadius: "5px",
                cursor: "pointer",
                fontSize: "1.1em",
                transition: "background-color 0.3s",
              }}
              onMouseOver={(e) => (e.target.style.backgroundColor = "#218838")}
              onMouseOut={(e) => (e.target.style.backgroundColor = "#28a745")}
            >
              Detect Them
            </button>
            <button
              onClick={handleClear}
              style={{
                padding: "12px 24px",
                backgroundColor: "#dc3545",
                color: "white",
                border: "none",
                borderRadius: "5px",
                cursor: "pointer",
                fontSize: "1.1em",
                transition: "background-color 0.3s",
                marginLeft: "20px",
              }}
              onMouseOver={(e) => (e.target.style.backgroundColor = "#c82333")}
              onMouseOut={(e) => (e.target.style.backgroundColor = "#dc3545")}
            >
              Clear
            </button>
          </div>
        )}
      </div>

      <div style={{ marginTop: "40px" }}>
        {labeledImages && (
          <div>
            <h2
              style={{ fontSize: "1.8em", color: "#333", textAlign: "center" }}
            >
              All Detected Images ({numImagesProcessed} processed)
            </h2>
            {["R-CNN", "YOLOv5", "YOLOv7", "YOLOv9"].map((algo) => (
              <div key={algo} style={{ marginBottom: "40px" }}>
                <h3 style={{ fontSize: "1.5em", color: "#444" }}>{algo}</h3>
                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: "10px",
                    justifyContent: "center",
                  }}
                >
                  {labeledImages[algo].map((imgUrl, index) => (
                    <img
                      key={index}
                      src={imgUrl}
                      alt={`${algo} detection ${index}`}
                      style={{
                        width: "200px",
                        height: "200px",
                        objectFit: "cover",
                        borderRadius: "5px",
                      }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {tableData.length > 0 && (
          <div style={{ marginTop: "40px" }}>
            <h2
              style={{ fontSize: "1.8em", color: "#333", textAlign: "center" }}
            >
              Comparison Table
            </h2>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                margin: "20px 0",
              }}
            >
              <thead>
                <tr style={{ backgroundColor: "#f2f2f2" }}>
                  {Object.keys(tableData[0]).map((key) => (
                    <th
                      key={key}
                      style={{
                        border: "1px solid #ddd",
                        padding: "10px",
                        textAlign: "center",
                        fontWeight: "bold",
                      }}
                    >
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
                    style={{
                      backgroundColor:
                        row.Algorithm === bestAlgo ? "#d4edda" : "transparent",
                    }}
                  >
                    {Object.entries(row).map(([key, value], i) => (
                      <td
                        key={i}
                        style={{
                          border: "1px solid #ddd",
                          padding: "10px",
                          textAlign: "center",
                        }}
                      >
                        {formatValue(key, value)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <h3
              style={{
                fontSize: "1.5em",
                color: "#28a745",
                textAlign: "center",
              }}
            >
              Best Algorithm: {bestAlgo}
            </h3>
            <div style={{ margin: "20px 0", textAlign: "left" }}>
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
                        backgroundColor: "#f9f9f9",
                        padding: "10px",
                        borderRadius: "5px",
                      }}
                    >
                      {explanation}
                    </pre>
                  </div>
                )
              )}
            </div>
            <div style={{ margin: "20px 0", textAlign: "left" }}>
              <h4 style={{ fontSize: "1.3em", color: "#444" }}>
                Why These Algorithms?
              </h4>
              <pre
                style={{
                  whiteSpace: "pre-wrap",
                  fontSize: "1em",
                  color: "#555",
                  backgroundColor: "#f9f9f9",
                  padding: "10px",
                  borderRadius: "5px",
                }}
              >
                {selectionReason}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
