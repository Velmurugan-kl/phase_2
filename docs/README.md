# Doctor-in-the-Loop Learning Framework

## 🎯 Overview

This is a comprehensive **Doctor-in-the-Loop Learning Framework** for continuous improvement of lung disease classification models. The system enables medical professionals to validate AI predictions, provide corrections, and automatically retrain models based on real-world feedback.

### Key Features

✅ **5-Class Lung Disease Classification**
- Bacterial Pneumonia
- Corona Virus Disease (COVID-19)
- Normal
- Tuberculosis
- Viral Pneumonia

✅ **Ensemble Model Architecture**
- YOLOv8 Classifier (30% weight)
- ResNet18 (50% weight)
- Vision Transformer (5% weight)
- Custom CNN (15% weight)

✅ **Doctor Feedback System**
- Interactive validation interface
- Correction recording
- Notes and annotations
- Doctor ID tracking

✅ **Automated Retraining Pipeline**
- Continuous learning from corrections
- Model fine-tuning
- Performance tracking
- Version control

✅ **Weight Optimization**
- Automatic ensemble weight adjustment
- Performance-based optimization
- A/B testing support

✅ **Comprehensive Analytics**
- Real-time performance dashboards
- Class-wise metrics
- Confusion matrices
- Trend analysis

---

## 📁 Project Structure

```
doctor_feedback_system/
├── app_enhanced.py              # Flask application with feedback system
├── ensemble.py                  # Original ensemble prediction logic
├── retrain_pipeline.py          # Automated retraining pipeline
├── optimize_weights.py          # Ensemble weight optimization
├── db_manager.py                # Database management utilities
│
├── templates/
│   ├── index.html               # Upload page (original)
│   ├── result_enhanced.html     # Results page with feedback UI
│   ├── dashboard.html           # Performance analytics dashboard
│   └── training_queue.html      # Training queue viewer
│
├── static/
│   ├── uploads/                 # User uploaded images
│   ├── feedback_images/         # Corrected images for retraining
│   └── training_export_*.json   # Exported training data
│
├── Models/
│   ├── best_yolo.pt            # YOLOv8 model
│   ├── best_resnet.pth         # ResNet18 model
│   ├── best_vit.pth            # Vision Transformer model
│   ├── best_cnn.h5             # Custom CNN model
│   └── retrained/              # Retrained model versions
│
└── doctor_feedback.db          # SQLite database
```

---

## 🗄️ Database Schema

### Tables

1. **predictions** - All model predictions
   - id, filename, image_path, predicted_class, confidence, all_probabilities, timestamp

2. **feedback** - Doctor validations
   - id, prediction_id, is_correct, correct_class, doctor_notes, feedback_timestamp, doctor_id

3. **training_queue** - Samples for retraining
   - id, feedback_id, image_path, true_class, priority, used_in_training

4. **model_performance** - Overall metrics
   - id, model_version, total_predictions, correct_predictions, accuracy

5. **class_performance** - Class-wise metrics
   - id, class_name, true_positives, false_positives, false_negatives, precision, recall, f1_score

---

## 🚀 Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# CUDA (optional, for GPU acceleration)
nvidia-smi
```

### Dependencies

```bash
# Install PyTorch (with CUDA if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install TensorFlow
pip install tensorflow

# Install Ultralytics YOLO
pip install ultralytics

# Install Flask and utilities
pip install flask
pip install pillow
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install scipy
pip install pydicom
pip install openpyxl
```

---

## 🎮 Usage

### 1. Start the Application

```bash
python app_enhanced.py
```

Access at: `http://localhost:5000`

### 2. Upload & Predict

1. Navigate to home page
2. Upload X-ray image (PNG, JPG, JPEG)
3. View AI predictions with confidence scores

### 3. Provide Feedback

**If prediction is correct:**
- Click "Prediction is Correct"

**If prediction is incorrect:**
- Click "Prediction is Incorrect"
- Select the correct diagnosis
- Add optional notes
- Submit correction

### 4. View Dashboard

Navigate to `/dashboard` to see:
- Overall accuracy
- Class-wise performance
- Recent feedback history
- Training queue status

### 5. Training Queue

Navigate to `/training_queue` to:
- View all corrected samples
- Export training data
- Monitor queue status

---

## 🔄 Retraining Workflow

### Automatic Retraining

```bash
# Run retraining pipeline
python retrain_pipeline.py
```

**What it does:**
1. Retrieves all unused feedback samples from database
2. Prepares training and validation datasets
3. Fine-tunes ResNet18, ViT, and CNN models
4. Saves retrained models with timestamps
5. Marks samples as used in training
6. Generates comprehensive report

**Minimum Requirements:**
- At least 10 feedback samples recommended
- Will skip if insufficient data

### Manual Retraining

```python
from retrain_pipeline import RetrainingPipeline

pipeline = RetrainingPipeline()
results = pipeline.run_full_pipeline(
    min_samples=10,  # Minimum samples needed
    epochs=10,       # Training epochs
    batch_size=8     # Batch size
)
```

---

## ⚖️ Weight Optimization

```bash
# Optimize ensemble weights based on feedback
python optimize_weights.py
```

**What it does:**
1. Loads all validated predictions
2. Runs individual models on each sample
3. Uses optimization algorithm to find best weights
4. Compares performance to baseline
5. Generates new weights file if improvement found

**Output:**
- `ensemble_optimized.py` - New weight configuration
- `weight_optimization_report_*.json` - Detailed analysis

**When to use optimized weights:**
- If improvement > 1%: Strongly recommended
- If improvement > 0.5%: Consider testing
- If improvement < 0.5%: Keep current weights

---

## 📊 Database Management

```bash
python db_manager.py
```

**Features:**
1. **View Statistics** - Comprehensive metrics
2. **Export to Excel** - All data in XLSX format
3. **Generate Visualizations** - Charts and graphs
4. **Backup Database** - Create timestamped backups
5. **Clean Old Data** - Remove old predictions without feedback

### Programmatic Access

```python
from db_manager import DatabaseManager

manager = DatabaseManager()

# Get statistics
stats = manager.get_statistics()

# Export data
manager.export_to_excel('report.xlsx')

# Create visualizations
manager.generate_visualizations()

# Backup database
manager.backup_database()
```

---

## 📈 API Endpoints

### GET `/`
Home page - Upload interface

### POST `/result`
Submit image for prediction
- **Input:** Image file (PNG/JPG/JPEG)
- **Output:** Predictions with confidence scores

### POST `/submit_feedback`
Submit doctor feedback
```json
{
  "prediction_id": 123,
  "is_correct": false,
  "correct_class": 2,
  "correct_class_name": "Normal",
  "doctor_notes": "Clear lungs, no abnormalities",
  "doctor_id": "DR001"
}
```

### GET `/dashboard`
Performance analytics dashboard

### GET `/training_queue`
View training queue

### GET `/export_training_data`
Export training queue as JSON

### POST `/predict_dicom`
DICOM file prediction
- **Input:** DICOM file (.dcm)
- **Output:** JSON with predictions

### GET `/api/stats`
Current statistics (JSON)

---

## 🔬 Model Performance Tracking

### Metrics Tracked

**Overall:**
- Total predictions
- Total feedback received
- Correct predictions
- Overall accuracy

**Per-Class:**
- True Positives (TP)
- False Positives (FP)
- False Negatives (FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 Score = 2 × (Precision × Recall) / (Precision + Recall)

### Real-time Updates

Metrics update automatically when:
- New feedback is submitted
- Models are retrained
- Weights are optimized

---

## 🔐 Security Considerations

1. **Authentication** - Add user authentication for production
2. **Data Privacy** - Anonymize patient data
3. **Access Control** - Role-based permissions
4. **Database Encryption** - Encrypt sensitive data
5. **Audit Logging** - Track all actions

**Recommendation:** Implement these before deployment in clinical settings.

---

## 🐛 Troubleshooting

### Database locked error
```bash
# Close all connections and restart
rm doctor_feedback.db-journal
python app_enhanced.py
```

### Models not loading
```bash
# Check model paths
ls -la Models/

# Verify models exist
python -c "import torch; print(torch.cuda.is_available())"
```

### Low accuracy after retraining
- Check if sufficient training samples (>20 recommended)
- Verify data quality
- Try increasing epochs
- Check for class imbalance

### Weight optimization shows no improvement
- Current weights may already be optimal
- Need more diverse feedback samples
- Consider adjusting individual model architectures

---

## 📋 Best Practices

### For Doctors

1. **Provide Clear Corrections**
   - Select the correct diagnosis carefully
   - Add detailed notes explaining the reasoning
   - Include relevant medical context

2. **Regular Validation**
   - Review predictions consistently
   - Prioritize uncertain cases (confidence < 80%)
   - Document edge cases

3. **Quality Over Quantity**
   - Focus on high-quality feedback
   - Flag ambiguous cases for discussion

### For Administrators

1. **Regular Retraining**
   - Run retraining when queue reaches 50+ samples
   - Schedule weekly/monthly retraining cycles
   - Monitor performance after each update

2. **Database Maintenance**
   - Backup database weekly
   - Clean old data quarterly
   - Export reports monthly

3. **Weight Optimization**
   - Run optimization after major retraining
   - Test optimized weights thoroughly before deployment
   - Keep baseline weights for rollback

---

## 📊 Performance Benchmarks

### Expected Metrics (with feedback)

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Overall Accuracy | >85% | >90% | >95% |
| Per-Class Precision | >80% | >85% | >90% |
| Per-Class Recall | >80% | >85% | >90% |
| F1 Score | >80% | >85% | >90% |

### Training Time Estimates

| Model | Samples | Epochs | GPU Time | CPU Time |
|-------|---------|--------|----------|----------|
| ResNet18 | 100 | 10 | ~5 min | ~20 min |
| ViT | 100 | 10 | ~8 min | ~35 min |
| CNN | 100 | 10 | ~3 min | ~12 min |
| YOLO | 100 | 10 | ~10 min | ~45 min |

---

## 🚀 Deployment

### Production Checklist

- [ ] Add user authentication
- [ ] Implement HTTPS
- [ ] Configure database backups
- [ ] Set up monitoring/logging
- [ ] Add rate limiting
- [ ] Implement error handling
- [ ] Create admin panel
- [ ] Document API thoroughly
- [ ] Set up CI/CD pipeline
- [ ] Configure load balancing (if needed)

### Environment Variables

```bash
export FLASK_ENV=production
export SECRET_KEY="your-secret-key"
export DATABASE_PATH="/path/to/db"
export MODELS_DIR="/path/to/models"
```

---

## 🤝 Contributing

### Adding New Disease Classes

1. Update `class_labels` in all scripts
2. Retrain all models with new classes
3. Update database schema
4. Modify frontend UI

### Adding New Models

1. Create model training script
2. Add prediction logic to ensemble
3. Update weight configuration
4. Test thoroughly before deployment

---

## 📄 License

This project is for educational and research purposes. Consult with legal/medical teams before clinical deployment.

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Review database statistics
3. Check model logs
4. Contact system administrator

---

## 🎯 Future Enhancements

### Planned Features

- [ ] Multi-language support
- [ ] Mobile app interface
- [ ] Real-time collaboration tools
- [ ] Advanced visualization (3D lung models)
- [ ] Integration with PACS systems
- [ ] Automated report generation
- [ ] Federated learning support
- [ ] Explainable AI (attention maps)
- [ ] 13-symptom classification (when clean dataset available)
- [ ] Doctor performance analytics
- [ ] A/B testing framework
- [ ] Model versioning system

---

## 📚 References

- YOLOv8: Ultralytics Documentation
- ResNet: Deep Residual Learning for Image Recognition
- Vision Transformer: An Image is Worth 16x16 Words
- Flask: Web Framework Documentation
- PyTorch: Deep Learning Framework
- TensorFlow/Keras: Machine Learning Library

---

**Last Updated:** 2024
**Version:** 2.0 (Phase 2 - Doctor-in-the-Loop Implementation)
