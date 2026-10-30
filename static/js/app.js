// FloraVision Frontend Interaction Controller
document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const loadingState = document.getElementById("loading-state");
  const emptyState = document.getElementById("empty-state");
  const resultsCard = document.getElementById("results-card");
  const tabButtons = document.querySelectorAll(".tab-btn");
  const plantButtons = document.querySelectorAll(".plant-btn");
  const subcategoryTray = document.getElementById("subcategory-tray");

  // Plant → Sub-category Disease Mapping
  const plantDiseases = {
    apple: [
      { label: "Apple Scab",   file: "Apple___Apple_scab_sample.jpg" },
      { label: "Black Rot",    file: "Apple___Black_rot_sample.jpg" },
      { label: "Cedar Rust",   file: "Apple___Cedar_apple_rust_sample.jpg" },
      { label: "Healthy",      file: "Apple___healthy_sample.jpg" }
    ],
    pepper: [
      { label: "Bacterial Spot", file: "Pepper,_bell___Bacterial_spot_sample.jpg" },
      { label: "Healthy",        file: "Pepper,_bell___healthy_sample.jpg" }
    ],
    cherry: [
      { label: "Powdery Mildew", file: "Cherry_(including_sour)___Powdery_mildew_sample.jpg" },
      { label: "Healthy",        file: "Cherry_(including_sour)___healthy_sample.jpg" }
    ],
    corn: [
      { label: "Gray Leaf Spot",    file: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot_sample.jpg" },
      { label: "Common Rust",       file: "Corn_(maize)___Common_rust__sample.jpg" },
      { label: "Northern Blight",   file: "Corn_(maize)___Northern_Leaf_Blight_sample.jpg" },
      { label: "Healthy",           file: "Corn_(maize)___healthy_sample.jpg" }
    ],
    grape: [
      { label: "Black Rot",    file: "Grape___Black_rot_sample.jpg" },
      { label: "Esca",         file: "Grape___Esca_(Black_Measles)_sample.jpg" },
      { label: "Leaf Blight",  file: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)_sample.jpg" },
      { label: "Healthy",      file: "Grape___healthy_sample.jpg" }
    ],
    peach: [
      { label: "Bacterial Spot", file: "Peach___Bacterial_spot_sample.jpg" },
      { label: "Healthy",        file: "Peach___healthy_sample.jpg" }
    ],
    potato: [
      { label: "Early Blight", file: "Potato___Early_blight_sample.jpg" },
      { label: "Late Blight",  file: "Potato___Late_blight_sample.jpg" },
      { label: "Healthy",      file: "Potato___healthy_sample.jpg" }
    ],
    strawberry: [
      { label: "Leaf Scorch", file: "Strawberry___Leaf_scorch_sample.jpg" },
      { label: "Healthy",     file: "Strawberry___healthy_sample.jpg" }
    ],
    tomato: [
      { label: "Bacterial Spot",  file: "Tomato___Bacterial_spot_sample.jpg" },
      { label: "Early Blight",    file: "Tomato___Early_blight_sample.jpg" },
      { label: "Late Blight",     file: "Tomato___Late_blight_sample.jpg" },
      { label: "Leaf Mold",       file: "Tomato___Leaf_Mold_sample.jpg" },
      { label: "Septoria Spot",   file: "Tomato___Septoria_leaf_spot_sample.jpg" },
      { label: "Spider Mites",    file: "Tomato___Spider_mites Two-spotted_spider_mite_sample.jpg" },
      { label: "Target Spot",     file: "Tomato___Target_Spot_sample.jpg" },
      { label: "Mosaic Virus",    file: "Tomato___Tomato_mosaic_virus_sample.jpg" },
      { label: "Yellow Curl",     file: "Tomato___Tomato_Yellow_Leaf_Curl_Virus_sample.jpg" },
      { label: "Healthy",         file: "Tomato___healthy_sample.jpg" }
    ]
  };

  // Plant button click → toggle sub-categories
  plantButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const plant = btn.getAttribute("data-plant");
      const wasActive = btn.classList.contains("active");

      // Deactivate all plant buttons
      plantButtons.forEach((b) => b.classList.remove("active"));

      if (wasActive) {
        // Collapse if same plant clicked again
        subcategoryTray.innerHTML = "";
        subcategoryTray.classList.remove("open");
        return;
      }

      // Activate clicked plant
      btn.classList.add("active");

      // Build sub-category chips
      const diseases = plantDiseases[plant] || [];
      subcategoryTray.innerHTML = "";
      diseases.forEach((d) => {
        const chip = document.createElement("button");
        chip.className = "sub-chip";
        chip.textContent = d.label;
        chip.addEventListener("click", () => analyzeSampleLeaf(d.file));
        subcategoryTray.appendChild(chip);
      });
      subcategoryTray.classList.add("open");
    });
  });

  // Drag and Drop Events
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  // Tab switching
  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      const targetTabId = btn.getAttribute("data-tab");
      document.querySelectorAll(".tab-content").forEach((content) => {
        content.style.display = content.id === targetTabId ? "block" : "none";
      });
    });
  });

  async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append("file", file);
    await sendAnalysisRequest("/api/analyze", {
      method: "POST",
      body: formData,
    });
  }

  async function analyzeSampleLeaf(sampleFilename) {
    await sendAnalysisRequest(`/api/analyze-sample?filename=${encodeURIComponent(sampleFilename)}`, {
      method: "GET",
    });
  }

  async function sendAnalysisRequest(url, options) {
    emptyState.style.display = "none";
    resultsCard.style.display = "none";
    loadingState.style.display = "block";

    try {
      const response = await fetch(url, options);
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      renderResults(data);
    } catch (err) {
      alert("Error analyzing image: " + err.message);
      emptyState.style.display = "block";
    } finally {
      loadingState.style.display = "none";
    }
  }

  function renderResults(data) {
    resultsCard.style.display = "block";

    const pred = data.prediction;
    const sev = data.severity;
    const vis = data.visualizations;

    // 1. HUD Cards (Health Condition & Affected Leaf Area)
    const hudStatus = document.getElementById("hud-status");
    hudStatus.textContent = pred.is_healthy ? "Healthy" : sev.health_status;
    hudStatus.className = "hud-value " + (pred.is_healthy ? "badge-healthy" : (sev.affected_area_pct > 20 ? "badge-danger" : "badge-warning"));

    const hudSeverity = document.getElementById("hud-severity");
    if (hudSeverity) {
      hudSeverity.textContent = `${sev.affected_area_pct}%`;
    }

    // 2. Diagnosis Banner
    const diagPlant = document.getElementById("diag-plant");
    if (diagPlant) diagPlant.textContent = pred.plant;

    const diagDisease = document.getElementById("diag-disease");
    if (diagDisease) diagDisease.textContent = pred.disease;

    const diagConfRate = document.getElementById("diag-conf-rate");
    if (diagConfRate) diagConfRate.textContent = `${pred.confidence_pct}% match`;

    // 3. Top-3 Prediction Bars
    const topContainer = document.getElementById("top-predictions-container");
    if (topContainer) {
      topContainer.innerHTML = "";
      pred.top3.forEach((item) => {
        const predDiv = document.createElement("div");
        predDiv.className = "pred-item";
        predDiv.innerHTML = `
          <div class="pred-info">
            <span>${item.label}</span>
            <span style="font-weight: 700;">${item.confidence}%</span>
          </div>
          <div class="pred-bar-bg">
            <div class="pred-bar-fill" style="width: ${item.confidence}%;"></div>
          </div>
        `;
        topContainer.appendChild(predDiv);
      });
    }

    // 4. Visual Panels
    document.getElementById("img-original").src = vis.original;
    document.getElementById("img-gradcam").src = vis.gradcam_overlay;
    document.getElementById("img-leaf-mask").src = vis.leaf_mask;
    document.getElementById("img-disease-mask").src = vis.disease_mask;
    document.getElementById("img-seg-overlay").src = vis.segmentation_overlay;
    document.getElementById("img-clahe").src = vis.clahe;
    document.getElementById("img-canny").src = vis.canny;
    document.getElementById("img-sobel").src = vis.sobel;
    document.getElementById("img-orb").src = vis.orb;
  }
});
