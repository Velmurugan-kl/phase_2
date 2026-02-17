# 🚀 Quick Start Guide - Doctor-in-the-Loop System

## ⚡ 5-Minute Setup

### Step 1: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all requirements
pip install -r requirements.txt
```

### Step 2: Verify Models

```bash
# Check if models exist
ls -la Ignore/Models/

# You should see:
# - best_yolo.pt
# - best_resnet.pth
# - best_vit.pth
# - best_cnn.h5
```

### Step 3: Initialize Database

```bash
# The database will auto-initialize on first run
# Or manually initialize:
python -c "from src.app_enhanced import init_db; init_db()"
```

### Step 4: Start Application

```bash
python src/app_enhanced.py
```

Visit: **http://localhost:5000**

---

## 🎯 First-Time Workflow

### 1️⃣ Upload an X-Ray

- Click "Upload image"
- Select PNG/JPG/JPEG file
- Click "Submit"

### 2️⃣ Review Prediction

- View AI predictions with confidence scores
- All 5 classes shown, sorted by confidence

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

### Run Retraining

```bash
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

✅ Optimized weights saved to: ensemble_optimized.py
```

**Next Steps:**

1. Review improvement (1.80% is significant!)
2. Update `ensemble.py` with new weights
3. Test on validation set
4. Deploy if results are better

---

## 📊 Database Management

### Quick Stats

```bash
python db_manager.py
# Select option 1 (View Statistics)
```

### Export Data

```bash
python db_manager.py
# Select option 2 (Export to Excel)
# File saved: feedback_report.xlsx
```

### Generate Visualizations

```bash
python db_manager.py
# Select option 3 (Generate Visualizations)
# Saved to: visualizations/
```

---

## 🔍 Common Tasks

### Check Current Performance

```python
from db_manager import DatabaseManager

manager = DatabaseManager()
stats = manager.get_statistics()

print(f"Accuracy: {stats['accuracy']:.2f}%")
print(f"Pending Training: {stats['pending_training']}")
```

### Export Training Data

```bash
curl http://localhost:5000/export_training_data
```

### View API Stats

```bash
curl http://localhost:5000/api/stats
```

Returns:

```json
{
  "total_predictions": 150,
  "total_feedback": 120,
  "correct_predictions": 105,
  "accuracy": 87.5,
  "pending_training_samples": 25
}
```

---

## 🐛 Troubleshooting

### Issue: "Models not found"

```bash
# Check Models directory
ls Models/

# If missing, ensure you have:
# - best_yolo.pt
# - best_resnet.pth
# - best_vit.pth
# - best_cnn.h5
```

### Issue: "Database is locked"

```bash
# Stop all running instances
pkill -f app_enhanced.py

# Remove lock file
rm doctor_feedback.db-journal

# Restart
python app_enhanced.py
```

### Issue: "Low accuracy after retraining"

**Possible causes:**

- Not enough training samples (need 50+)
- Class imbalance
- Need more epochs

**Solutions:**

```bash
# Collect more feedback
# Then run with more epochs:
python -c "
from retrain_pipeline import RetrainingPipeline
pipeline = RetrainingPipeline()
pipeline.run_full_pipeline(min_samples=10, epochs=20, batch_size=8)
"
```

---

## 📋 Daily Workflow

### For Doctors

**Morning:**

1. Check dashboard for accuracy
2. Review overnight predictions

**Throughout Day:** 3. Upload X-rays as needed 4. Validate predictions immediately 5. Add detailed notes for corrections

**End of Day:** 6. Review today's feedback 7. Check training queue

### For Administrators

**Weekly:**

1. Backup database
2. Review performance trends
3. Run retraining if queue > 50

**Monthly:**

1. Export comprehensive report
2. Generate visualizations
3. Optimize ensemble weights
4. Clean old data

---

## 🎓 Training Tips

### Maximize Model Performance

1. **Diverse Feedback**
   - Get feedback from multiple doctors
   - Cover all disease classes
   - Include edge cases

2. **Quality Corrections**
   - Add detailed notes
   - Explain reasoning
   - Flag uncertain cases

3. **Regular Retraining**
   - Weekly if high volume
   - Monthly for lower volume
   - After major corrections

4. **Monitor Metrics**
   - Watch per-class accuracy
   - Check for drift
   - Compare before/after retraining

---

## 🚀 Advanced Usage

### Custom Retraining Schedule

```python
from retrain_pipeline import RetrainingPipeline
import schedule
import time

def scheduled_retrain():
    pipeline = RetrainingPipeline()
    pipeline.run_full_pipeline(min_samples=20, epochs=15)

# Every Sunday at 2 AM
schedule.every().sunday.at("02:00").do(scheduled_retrain)

while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour
```

### Automated Weight Optimization

```python
from optimize_weights import WeightOptimizer

optimizer = WeightOptimizer()
results = optimizer.run_optimization()

# Auto-update if improvement > 1%
if results and results['improvement'] > 1.0:
    print("Significant improvement! Updating weights...")
    # Copy ensemble_optimized.py to ensemble.py
    import shutil
    shutil.copy('ensemble_optimized.py', 'ensemble.py')
```

### Batch Prediction

```python
import os
from app_enhanced import ensemble_predict

image_dir = "batch_images/"
for filename in os.listdir(image_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, filename)
        final_class, final_probs = ensemble_predict(
            image_path, yolo_model, resnet_model,
            vit_model, cnn_model, transform, device, weights
        )
        print(f"{filename}: {class_labels[final_class]} ({final_probs[final_class]*100:.2f}%)")
```

---

## 📞 Getting Help

1. **Check README.md** - Comprehensive documentation
2. **Run db_manager.py** - View system statistics
3. **Check logs** - Application console output
4. **Review dashboard** - Performance metrics

---

## 🎯 Success Metrics

Track these to measure system improvement:

- [ ] Accuracy increasing over time
- [ ] Consistent feedback collection
- [ ] Regular retraining cycles
- [ ] Balanced class distribution
- [ ] Doctor engagement
- [ ] Reduced prediction errors

**Goal:** 90%+ accuracy with 100+ feedback samples

---

**You're all set! 🎉**

Start with a few predictions, provide feedback, and watch your AI improve!
