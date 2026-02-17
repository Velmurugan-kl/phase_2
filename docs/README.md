# Doctor-in-the-Loop Learning Framework
### Lung Disease Classification with Continuous Learning

A medical AI system where doctors validate predictions and corrections are used to retrain the ensemble model over time.

---

## What It Does

Classifies chest X-rays into 5 categories using an ensemble of 4 models:

- Bacterial Pneumonia
- Corona Virus Disease (COVID-19)
- Normal
- Tuberculosis
- Viral Pneumonia

Doctors review each prediction, mark it correct or provide the right diagnosis. Those corrections are stored and used to retrain the models, making the system progressively more accurate.

---

## Project Structure

```
doctor_feedback_system/
├── src/
│   ├── app_enhanced.py          # Main Flask application
│   ├── ensemble.py              # Ensemble prediction logic
│   ├── retrain_pipeline.py      # Retraining pipeline
│   ├── optimize_weights.py      # Weight optimization
│   ├── db_manager.py            # Database utilities
│   ├── config.py                # Configuration
│   ├── templates/               # HTML pages
│   └── static/                  # Uploads and feedback images
│       ├── uploads/
│       └── feedback_images/
├── Models/                      # Place downloaded models here (inside src/)
├── doctor_feedback.db           # Auto-generated SQLite database
└── pyproject.toml
```

---

## Setup Instructions

### Prerequisites

- Python 3.10 or 3.11
- [Poetry](https://python-poetry.org/docs/#installation)

Install Poetry if you don't have it:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

---

### 1. Clone the Repository

```bash
git clone <repository-url>
cd doctor_feedback_system
```

---

### 2. Download the Models

Download the pre-trained models from the link below and place the entire `Models` folder inside the `src/` directory:

📁 **[Download Models Folder](https://drive.google.com/your-link-here)**

After downloading, your structure should look like:

```
src/
└── Models/
    ├── best_yolo.pt
    ├── best_resnet.pth
    ├── best_vit.pth
    └── best_cnn.h5
```

---

### 3. Install Dependencies

```bash
poetry install
```

This creates a `.venv` folder directly inside the project directory (configured via `poetry.toml`) and installs all required packages into it.

---

### 4. Activate the Virtual Environment

```bash
poetry shell
```

You should see your terminal prompt change indicating the environment is active. The virtual environment lives at `.venv/` in the project root, so you can also activate it manually if needed:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

---

### 5. Run the Application

```bash
cd src
poetry run python app_enhanced.py
```

Or using the configured CLI command:

```bash
poetry run lung-app
```

Open your browser and go to: **http://localhost:5000**

---

## Using the System

**Upload & Predict** — Upload a chest X-ray (PNG, JPG, or JPEG) from the home page. The ensemble model returns predictions with confidence scores for all 5 classes.

**Doctor Validation** — On the results page, mark the prediction as correct or select the right diagnosis if it's wrong. You can add notes and a doctor ID.

**Dashboard** — Visit `/dashboard` to see overall accuracy, per-class precision/recall/F1, and recent feedback history.

**Training Queue** — Visit `/training_queue` to review all corrections waiting to be used for retraining.

---

## Retraining the Models

Once enough corrections have been collected (minimum 10 recommended), run the retraining pipeline:

```bash
poetry run lung-retrain
```

This fine-tunes ResNet18, ViT, and CNN on the corrected samples and saves new model weights to `src/Models/retrained/`.

To optimize ensemble weights based on validated predictions:

```bash
poetry run lung-optimize
```

---

## Other Useful Commands

```bash
poetry run lung-db        # Database management (export, backup, visualizations)
poetry run lung-config    # View and validate configuration
```

---

## Troubleshooting

**Models not found** — Ensure the `Models/` folder is placed inside `src/` and contains all four files: `best_yolo.pt`, `best_resnet.pth`, `best_vit.pth`, `best_cnn.h5`.

**Poetry not recognized** — Restart your terminal after installing Poetry and ensure `~/.local/bin` is in your PATH.

**Database errors** — Delete `doctor_feedback.db` and restart the app. It will be recreated automatically.

**Insufficient samples for retraining** — At least 5 corrections are needed. The dashboard shows how many are currently queued.

---

## Notes

- This project is for educational and research purposes only
- Do not deploy in a clinical setting without proper authentication, HTTPS, and regulatory compliance
- Model weights are not included in the repository due to file size — download separately from the link above