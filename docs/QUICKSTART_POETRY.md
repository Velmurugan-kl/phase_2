# 🚀 Quick Start Guide - Poetry Edition

## ⚡ 5-Minute Setup with Poetry

### What is Poetry?

Poetry is a modern Python dependency management tool that:
- ✅ Manages dependencies automatically
- ✅ Creates isolated virtual environments
- ✅ Handles version conflicts
- ✅ Makes deployment easier
- ✅ Provides CLI commands

---

## 📦 Installation Methods

### Option 1: Poetry Setup (Recommended)

#### Step 1: Install Poetry (if not already installed)

**Windows (PowerShell):**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

**Linux/macOS/WSL:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Verify Installation:**
```bash
poetry --version
```

#### Step 2: Run Poetry Setup

```bash
# Run the automated setup wizard
python poetry_setup.py
```

The wizard will:
1. ✅ Check Python version (3.8+ required)
2. ✅ Verify Poetry installation
3. ✅ Create directory structure
4. ✅ Install all dependencies
5. ✅ Initialize database
6. ✅ Create configuration files

#### Step 3: Add Your Models

Place these in `Models/` directory:
- `best_yolo.pt`
- `best_resnet.pth`
- `best_vit.pth`
- `best_cnn.h5`

#### Step 4: Start the Application

```bash
# Activate Poetry virtual environment
poetry shell

# Start the app
poetry run python app_enhanced.py

# OR use the CLI command
poetry run lung-app
```

Visit: **http://localhost:5000**

---

### Option 2: Manual Poetry Setup

```bash
# 1. Install dependencies
poetry install

# 2. Create directories
mkdir -p Models/retrained static/uploads static/feedback_images templates backups visualizations

# 3. Add your models to Models/

# 4. Initialize database
poetry run python -c "from app_enhanced import init_db; init_db()"

# 5. Start app
poetry shell
python app_enhanced.py
```

---

## 🎯 Poetry Commands Reference

### Basic Commands

```bash
# Install all dependencies
poetry install

# Activate virtual environment
poetry shell

# Run commands without activating shell
poetry run python app_enhanced.py

# Add a new dependency
poetry add package-name

# Update dependencies
poetry update

# Show installed packages
poetry show

# Exit virtual environment
exit  # or deactivate
```

### Application Commands

Poetry provides convenient CLI commands:

```bash
# Start main application
poetry run lung-app

# Run retraining pipeline
poetry run lung-retrain

# Optimize ensemble weights
poetry run lung-optimize

# Database management
poetry run lung-db

# Configuration management
poetry run lung-config
```

---

## 🎮 First-Time Workflow

### 1️⃣ Upload an X-Ray

**Option A: Web Interface**
```
1. Navigate to http://localhost:5000
2. Click "Upload image"
3. Select PNG/JPG/JPEG file
4. Click "Submit"
```

**Option B: Command Line (for testing)**
```bash
poetry shell
python -c "
from app_enhanced import ensemble_predict, yolo_model, resnet_model, vit_model, cnn_model, transform, device, weights
final_class, final_probs = ensemble_predict('test_image.png', yolo_model, resnet_model, vit_model, cnn_model, transform, device, weights)
print(f'Predicted class: {final_class}, Confidence: {final_probs[final_class]*100:.2f}%')
"
```

### 2️⃣ Review Prediction

View AI predictions with confidence scores for all 5 classes

### 3️⃣ Provide Feedback

**If correct:**
```
✅ Click "Prediction is Correct"
```

**If incorrect:**
```
❌ Click "Prediction is Incorrect"
📝 Select correct diagnosis from list
💬 Add notes (optional)
📤 Click "Submit Correction"
```

### 4️⃣ View Dashboard
```
Navigate to: http://localhost:5000/dashboard

See:
- Total predictions
- Accuracy rate
- Class-wise performance
- Recent feedback
```

---

## 🔄 Your First Retraining

### When to Retrain?
- **Minimum:** 10 corrected samples
- **Recommended:** 50+ corrected samples
- **Optimal:** 100+ corrected samples

### Run Retraining with Poetry

```bash
# Activate environment
poetry shell

# Run retraining pipeline
poetry run lung-retrain

# OR directly
python retrain_pipeline.py
```

**Expected Output:**
```
🚀 STARTING AUTOMATED RETRAINING PIPELINE
📊 Retrieved 25 feedback samples for retraining
✅ Found 25 samples - proceeding with retraining

🔄 Retraining ResNet18...
Epoch [1/10], Loss: 0.5234, Val Acc: 85.00%
Epoch [2/10], Loss: 0.3421, Val Acc: 90.00%
...
✅ Best model saved with Val Acc: 92.00%

✅ RETRAINING PIPELINE COMPLETED SUCCESSFULLY
```

---

## ⚖️ Optimize Ensemble Weights

### When to Optimize?
- After collecting 20+ validated predictions
- After significant retraining
- When accuracy plateaus

### Run Optimization

```bash
poetry shell
poetry run lung-optimize

# OR
python optimize_weights.py
```

**Sample Output:**
```
🎯 Running optimization...

📊 OPTIMIZATION RESULTS
🔹 Baseline Accuracy (current weights): 87.50%
🔹 Optimized Accuracy: 89.30%
🔹 Improvement: 1.80%

📊 Weight Comparison:
Model      Current      Optimized    Change      
--------------------------------------------------
YOLO       0.300        0.250        -0.050
ResNet     0.500        0.550        +0.050
ViT        0.050        0.075        +0.025
CNN        0.150        0.125        -0.025
```

---

## 📊 Database Management with Poetry

### Quick Stats

```bash
poetry shell
poetry run lung-db
# Select option 1 (View Statistics)
```

### Export Data

```bash
poetry run lung-db
# Select option 2 (Export to Excel)
# File saved: feedback_report.xlsx
```

### Generate Visualizations

```bash
poetry run lung-db
# Select option 3 (Generate Visualizations)
# Saved to: visualizations/
```

---

## 🔍 Common Tasks with Poetry

### Development Workflow

```bash
# Start development session
poetry shell

# Run app in debug mode (default in config.json)
python app_enhanced.py

# Open another terminal for parallel tasks
poetry run lung-db  # Check stats
poetry run lung-retrain  # Retrain models
```

### Adding New Dependencies

```bash
# Add a package
poetry add scikit-image

# Add a dev dependency
poetry add --group dev jupyterlab

# Update all dependencies
poetry update

# Remove a package
poetry remove package-name
```

### Production Deployment

```bash
# Export requirements.txt for production
poetry export -f requirements.txt --output requirements.txt --without-hashes

# Install in production (without dev dependencies)
poetry install --no-dev

# Build distributable package
poetry build
```

---

## 🐛 Troubleshooting

### Issue: "Poetry not found"

```bash
# Verify installation
poetry --version

# If not found, add to PATH (example for macOS/Linux)
export PATH="$HOME/.local/bin:$PATH"

# For Windows, add to PATH:
# C:\Users\YourUsername\AppData\Roaming\Python\Scripts
```

### Issue: "Python version mismatch"

```bash
# Check current Python version
python --version

# Tell Poetry to use specific Python version
poetry env use python3.9

# Or use full path
poetry env use /usr/bin/python3.9
```

### Issue: "Dependency conflicts"

```bash
# Clear cache
poetry cache clear pypi --all

# Remove lock file and reinstall
rm poetry.lock
poetry install
```

### Issue: "Models not loading"

```bash
# Check Models directory
ls -la Models/

# Verify in Python
poetry run python -c "
import os
models = ['best_yolo.pt', 'best_resnet.pth', 'best_vit.pth', 'best_cnn.h5']
for m in models:
    print(f'{m}: {os.path.exists(f\"Models/{m}\")}')"
```

### Issue: "Database locked"

```bash
# Stop all running instances
pkill -f app_enhanced.py

# Remove lock file
rm doctor_feedback.db-journal

# Restart
poetry run lung-app
```

---

## 📋 Daily Workflow with Poetry

### For Doctors

**Morning:**
```bash
poetry shell
python app_enhanced.py
# Open http://localhost:5000/dashboard
```

**Throughout Day:**
1. Upload X-rays as needed
2. Validate predictions immediately
3. Add detailed notes for corrections

**End of Day:**
```bash
# Check dashboard
curl http://localhost:5000/api/stats

# Or visit dashboard in browser
```

### For Administrators

**Weekly:**
```bash
# 1. Check stats
poetry run lung-db  # Option 1

# 2. Backup database
poetry run lung-db  # Option 4

# 3. Check training queue
poetry shell
python -c "
import sqlite3
conn = sqlite3.connect('doctor_feedback.db')
c = conn.cursor()
c.execute('SELECT COUNT(*) FROM training_queue WHERE used_in_training = 0')
print(f'Pending samples: {c.fetchone()[0]}')
conn.close()"

# 4. Retrain if needed (50+ samples)
poetry run lung-retrain
```

**Monthly:**
```bash
# 1. Generate reports
poetry run lung-db  # Options 2 & 3

# 2. Optimize weights
poetry run lung-optimize

# 3. Review performance trends
```

---

## 🎓 Poetry Best Practices

### Virtual Environments

```bash
# Show current environment info
poetry env info

# List all environments
poetry env list

# Remove environment
poetry env remove python3.9

# Create new environment
poetry env use python3.9
```

### Dependency Management

```bash
# Lock dependencies without installing
poetry lock

# Install only production dependencies
poetry install --no-dev

# Show dependency tree
poetry show --tree

# Check for security vulnerabilities
poetry show --outdated
```

### Scripts and Commands

Defined in `pyproject.toml`:
```toml
[tool.poetry.scripts]
lung-app = "app_enhanced:main"
lung-retrain = "retrain_pipeline:main"
lung-optimize = "optimize_weights:main"
lung-db = "db_manager:main"
lung-config = "config:main"
```

Use them:
```bash
poetry run lung-app        # Start application
poetry run lung-retrain    # Run retraining
poetry run lung-optimize   # Optimize weights
poetry run lung-db         # Database tools
poetry run lung-config     # Configuration
```

---

## 🚀 Quick Reference Card

```bash
# SETUP
poetry install              # Install dependencies
poetry shell               # Activate environment

# RUN APPLICATION
poetry run lung-app        # Start web app
poetry run lung-retrain    # Retrain models
poetry run lung-optimize   # Optimize weights
poetry run lung-db         # Database management

# DEVELOPMENT
poetry add <package>       # Add dependency
poetry update             # Update all dependencies
poetry show               # Show installed packages

# ENVIRONMENT
poetry env info           # Show environment info
poetry env list           # List environments
exit                      # Deactivate environment

# EXPORT
poetry export -f requirements.txt --output requirements.txt
```

---

## 📊 Comparison: pip vs Poetry

| Task | pip + venv | Poetry |
|------|-----------|--------|
| Install deps | `pip install -r requirements.txt` | `poetry install` |
| Add package | `pip install X; pip freeze > requirements.txt` | `poetry add X` |
| Activate env | `source venv/bin/activate` | `poetry shell` |
| Run script | `python script.py` | `poetry run script` |
| Update deps | Manual | `poetry update` |
| Resolve conflicts | Manual | Automatic |

---

## ✅ Success Checklist

- [ ] Poetry installed and working
- [ ] Dependencies installed (`poetry install`)
- [ ] All 4 models in `Models/` directory
- [ ] Database initialized
- [ ] Templates in `templates/` folder
- [ ] Can run `poetry run lung-app` successfully
- [ ] Can access http://localhost:5000
- [ ] Dashboard accessible at /dashboard

---

**You're all set with Poetry! 🎉**

Poetry makes dependency management effortless and deployment reliable. Your Doctor-in-the-Loop system is now ready for development and production use!

---

## 🔗 Useful Links

- **Poetry Documentation:** https://python-poetry.org/docs/
- **PyPI:** https://pypi.org/
- **Project README:** See README.md for full documentation
- **Implementation Details:** See IMPLEMENTATION_SUMMARY.md
