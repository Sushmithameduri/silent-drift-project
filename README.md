# Silent Drift Detection 🚦  
_Efficient, Lightweight Anomaly Detection for ML Telemetry_

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) 
![License](https://img.shields.io/badge/license-MIT-blue) 
![Version](https://img.shields.io/badge/version-1.0.0-orange)

---

## Description

Silent Drift Detection is a Python-based pipeline for detecting "silent drift" in machine learning telemetry using robust multivariate statistical methods. It helps you monitor your ML systems for subtle, hard-to-detect changes in data distribution—before they impact model performance.

**Why is this useful?**  
Silent drift can degrade model accuracy without obvious failures. This tool provides early warning, enabling proactive retraining and system health monitoring.

---

## Key Features

- 🚀 **End-to-End Pipeline:** From synthetic data generation to evaluation and visualization.
- 📊 **Mahalanobis Distance Detector:** Multivariate anomaly scoring for robust drift detection.
- 🛠️ **Rolling-Window Feature Engineering:** Captures evolving data patterns.
- 🖼️ **Automated Visualizations:** Plots anomaly scores and detection thresholds.
- 🧪 **Test Suite:** Includes unit tests for reliability.
- ⚡ **Lightweight & Fast:** Minimal dependencies, easy to integrate.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Sushmithameduri/silent-drift-project.git
cd silent-drift-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

Run the full pipeline:

```bash
python main.py
```

This will:
- Generate synthetic telemetry data
- Compute rolling features
- Fit the anomaly detector
- Score the monitoring period
- Save results and plots to `data/processed/`

### Example Output

![Anomaly Scores Over Time](data/processed/anomaly_scores.png)

---

## Configuration

You can customize the pipeline via command-line arguments in `main.py`:

- `--n`: Number of samples (default: 1000)
- `--seed`: Random seed for reproducibility
- `--train`: Train the detector
- `--eval`: Evaluate and plot results
- `--output-dir`: Output directory for artifacts

Example:

```bash
python main.py --n 2000 --seed 123 --output-dir ./output
```

---

## Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please see the `CONTRIBUTING.md` for guidelines.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

_Questions or feedback? Open an issue or contact the maintainer!_


