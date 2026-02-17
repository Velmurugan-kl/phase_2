# ---------------- Imports ----------------
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from torchvision import models, transforms
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
from ensemble import ensemble_predict, weights
import sqlite3
from datetime import datetime
import json
import shutil

# ---------------- Flask App ----------------
app = Flask(__name__)

# Set base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, "src", "static", "uploads")
app.config['FEEDBACK_FOLDER'] = os.path.join(BASE_DIR, "src", "static", "feedback_images")
app.config['DATABASE'] = os.path.join(BASE_DIR, "doctor_feedback.db")
app.secret_key = "secret key"

# Set template and static folders
app.template_folder = os.path.join(BASE_DIR, "src", "templates")
app.static_folder = os.path.join(BASE_DIR, "src", "static")

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FEEDBACK_FOLDER'], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Database Setup ----------------
def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    
    # Predictions table - stores all predictions
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT NOT NULL,
                  image_path TEXT NOT NULL,
                  predicted_class INTEGER NOT NULL,
                  predicted_class_name TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  all_probabilities TEXT NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  model_version TEXT DEFAULT 'v1.0')''')
    
    # Feedback table - stores doctor corrections
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  prediction_id INTEGER NOT NULL,
                  is_correct BOOLEAN NOT NULL,
                  correct_class INTEGER,
                  correct_class_name TEXT,
                  doctor_notes TEXT,
                  feedback_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  doctor_id TEXT,
                  FOREIGN KEY (prediction_id) REFERENCES predictions(id))''')
    
    # Training queue - images marked for retraining
    c.execute('''CREATE TABLE IF NOT EXISTS training_queue
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  feedback_id INTEGER NOT NULL,
                  image_path TEXT NOT NULL,
                  true_class INTEGER NOT NULL,
                  true_class_name TEXT NOT NULL,
                  priority INTEGER DEFAULT 1,
                  used_in_training BOOLEAN DEFAULT 0,
                  added_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (feedback_id) REFERENCES feedback(id))''')
    
    # Model performance tracking
    c.execute('''CREATE TABLE IF NOT EXISTS model_performance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  model_version TEXT NOT NULL,
                  total_predictions INTEGER DEFAULT 0,
                  correct_predictions INTEGER DEFAULT 0,
                  accuracy REAL DEFAULT 0.0,
                  last_updated DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Class-wise performance
    c.execute('''CREATE TABLE IF NOT EXISTS class_performance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  class_id INTEGER NOT NULL,
                  class_name TEXT NOT NULL,
                  true_positives INTEGER DEFAULT 0,
                  false_positives INTEGER DEFAULT 0,
                  false_negatives INTEGER DEFAULT 0,
                  precision REAL DEFAULT 0.0,
                  recall REAL DEFAULT 0.0,
                  f1_score REAL DEFAULT 0.0,
                  last_updated DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# ---------------- Load Models ----------------
def _find_model_file(filename: str):
    """Search candidate model directories and return absolute path if found."""
    candidate_dirs = [
        os.path.join(BASE_DIR, "Ignore", "Models"),
        os.path.join(BASE_DIR, "Models"),
        os.path.join(os.path.dirname(__file__), "Models"),
    ]
    looked = []
    for d in candidate_dirs:
        path = os.path.join(d, filename)
        looked.append(path)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Model file '{filename}' not found. Looked in: {', '.join(looked)}"
    )


# Load models (searching multiple possible locations)
yolo_path = _find_model_file("best_yolo.pt")
yolo_model = YOLO(yolo_path)

resnet_model = models.resnet18()
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 5)
resnet_path = _find_model_file("best_resnet.pth")
resnet_model.load_state_dict(torch.load(resnet_path, map_location=device))
resnet_model.to(device).eval()

vit_model = models.vit_b_16()
vit_model.heads.head = nn.Linear(vit_model.heads.head.in_features, 5)
vit_path = _find_model_file("best_vit.pth")
vit_model.load_state_dict(torch.load(vit_path, map_location=device))
vit_model.to(device).eval()

cnn_path = _find_model_file("best_cnn.h5")
cnn_model = load_model(cnn_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class_labels = ["Bacterial Pneumonia", "Corona Virus Disease", "Normal", "Tuberculosis", "Viral Pneumonia"]

# ---------------- Helper Functions ----------------
def save_prediction_to_db(filename, image_path, predicted_class, predicted_class_name, 
                         confidence, all_probabilities):
    """Save prediction to database"""
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    
    c.execute('''INSERT INTO predictions 
                 (filename, image_path, predicted_class, predicted_class_name, 
                  confidence, all_probabilities)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (filename, image_path, predicted_class, predicted_class_name,
               confidence, json.dumps(all_probabilities)))
    
    prediction_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return prediction_id

def save_feedback_to_db(prediction_id, is_correct, correct_class=None, 
                       correct_class_name=None, doctor_notes=None, doctor_id=None):
    """Save doctor feedback to database"""
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    
    c.execute('''INSERT INTO feedback 
                 (prediction_id, is_correct, correct_class, correct_class_name, 
                  doctor_notes, doctor_id)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (prediction_id, is_correct, correct_class, correct_class_name,
               doctor_notes, doctor_id))
    
    feedback_id = c.lastrowid
    
    # If incorrect, add to training queue
    if not is_correct and correct_class is not None:
        # Get image path from predictions
        c.execute('SELECT image_path FROM predictions WHERE id = ?', (prediction_id,))
        image_path = c.fetchone()[0]
        
        # Copy image to feedback folder for retraining
        feedback_image_path = os.path.join(
            app.config['FEEDBACK_FOLDER'], 
            f"{feedback_id}_{os.path.basename(image_path)}"
        )
        shutil.copy2(image_path, feedback_image_path)
        
        c.execute('''INSERT INTO training_queue 
                     (feedback_id, image_path, true_class, true_class_name, priority)
                     VALUES (?, ?, ?, ?, ?)''',
                  (feedback_id, feedback_image_path, correct_class, correct_class_name, 2))
    
    conn.commit()
    conn.close()
    
    # Update performance metrics
    update_performance_metrics()
    
    return feedback_id

def update_performance_metrics():
    """Update model performance metrics based on feedback"""
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    
    # Get total predictions with feedback
    c.execute('''SELECT COUNT(*) FROM feedback''')
    total_with_feedback = c.fetchone()[0]
    
    if total_with_feedback == 0:
        conn.close()
        return
    
    # Get correct predictions
    c.execute('''SELECT COUNT(*) FROM feedback WHERE is_correct = 1''')
    correct = c.fetchone()[0]
    
    accuracy = (correct / total_with_feedback) * 100 if total_with_feedback > 0 else 0
    
    # Update or insert overall performance
    c.execute('''INSERT OR REPLACE INTO model_performance 
                 (id, model_version, total_predictions, correct_predictions, accuracy, last_updated)
                 VALUES (1, 'v1.0', ?, ?, ?, ?)''',
              (total_with_feedback, correct, accuracy, datetime.now()))
    
    # Update class-wise performance
    for class_id, class_name in enumerate(class_labels):
        # True Positives: Predicted X and is correct
        c.execute('''SELECT COUNT(*) FROM predictions p
                     JOIN feedback f ON p.id = f.prediction_id
                     WHERE p.predicted_class = ? AND f.is_correct = 1''', (class_id,))
        tp = c.fetchone()[0]
        
        # False Positives: Predicted X but actually Y
        c.execute('''SELECT COUNT(*) FROM predictions p
                     JOIN feedback f ON p.id = f.prediction_id
                     WHERE p.predicted_class = ? AND f.is_correct = 0''', (class_id,))
        fp = c.fetchone()[0]
        
        # False Negatives: Predicted Y but actually X
        c.execute('''SELECT COUNT(*) FROM feedback
                     WHERE correct_class = ? AND is_correct = 0''', (class_id,))
        fn = c.fetchone()[0]
        
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        c.execute('''INSERT OR REPLACE INTO class_performance 
                     (id, class_id, class_name, true_positives, false_positives, 
                      false_negatives, precision, recall, f1_score, last_updated)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (class_id + 1, class_id, class_name, tp, fp, fn, 
                   precision, recall, f1, datetime.now()))
    
    conn.commit()
    conn.close()

# ---------------- Routes ----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def resultc():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash('Image successfully uploaded and displayed below')

        # Run ensemble
        final_class, final_probs = ensemble_predict(
            file_path, yolo_model, resnet_model, vit_model, cnn_model,
            transform, device, weights
        )
        
        # Create list of all predictions sorted by confidence
        predictions = []
        for i, prob in enumerate(final_probs):
            predictions.append({
                'class_name': class_labels[i],
                'class_id': i,
                'confidence': prob * 100
            })
        
        # Sort predictions by confidence in descending order
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        # Get top prediction details
        pred_class_name = class_labels[final_class]
        pred_class_likelihood = final_probs[final_class] * 100
        
        # Save to database
        prediction_id = save_prediction_to_db(
            filename, file_path, final_class, pred_class_name,
            pred_class_likelihood, final_probs.tolist()
        )

        return render_template(
            'result_enhanced.html',
            filename=filename,
            prediction_id=prediction_id,
            r=final_class,
            pred_class_name=pred_class_name,
            pred_likelihood=pred_class_likelihood,
            predictions=predictions,
            class_labels=class_labels
        )

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """Handle doctor feedback submission"""
    data = request.json
    
    prediction_id = data.get('prediction_id')
    is_correct = data.get('is_correct')
    correct_class = data.get('correct_class')
    correct_class_name = data.get('correct_class_name')
    doctor_notes = data.get('doctor_notes', '')
    doctor_id = data.get('doctor_id', 'anonymous')
    
    feedback_id = save_feedback_to_db(
        prediction_id, is_correct, correct_class, 
        correct_class_name, doctor_notes, doctor_id
    )
    
    return jsonify({
        'success': True,
        'feedback_id': feedback_id,
        'message': 'Feedback recorded successfully. Thank you for improving the system!'
    })

@app.route('/dashboard')
def dashboard():
    """Display doctor dashboard with performance metrics"""
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Get overall performance
    c.execute('SELECT * FROM model_performance WHERE id = 1')
    overall_perf = c.fetchone()
    
    # Get class-wise performance
    c.execute('SELECT * FROM class_performance ORDER BY class_id')
    class_perf = c.fetchall()
    
    # Get recent feedback
    c.execute('''SELECT p.filename, p.predicted_class_name, p.confidence,
                        f.is_correct, f.correct_class_name, f.doctor_notes,
                        f.feedback_timestamp
                 FROM feedback f
                 JOIN predictions p ON f.prediction_id = p.id
                 ORDER BY f.feedback_timestamp DESC
                 LIMIT 20''')
    recent_feedback = c.fetchall()
    
    # Get training queue count
    c.execute('SELECT COUNT(*) FROM training_queue WHERE used_in_training = 0')
    queue_count = c.fetchone()[0]
    
    conn.close()
    
    return render_template(
        'dashboard.html',
        overall_perf=overall_perf,
        class_perf=class_perf,
        recent_feedback=recent_feedback,
        queue_count=queue_count,
        class_labels=class_labels
    )

@app.route('/training_queue')
def training_queue():
    """Display images in training queue"""
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('''SELECT tq.*, f.doctor_notes, p.predicted_class_name, p.confidence
                 FROM training_queue tq
                 JOIN feedback f ON tq.feedback_id = f.id
                 JOIN predictions p ON f.prediction_id = p.id
                 WHERE tq.used_in_training = 0
                 ORDER BY tq.priority DESC, tq.added_timestamp ASC''')
    
    queue_items = c.fetchall()
    conn.close()
    
    return render_template('training_queue.html', queue_items=queue_items)

@app.route('/export_training_data')
def export_training_data():
    """Export training queue data for retraining"""
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('''SELECT * FROM training_queue WHERE used_in_training = 0''')
    items = c.fetchall()
    
    # Prepare export data
    export_data = {
        'total_samples': len(items),
        'export_timestamp': datetime.now().isoformat(),
        'samples': []
    }
    
    for item in items:
        export_data['samples'].append({
            'id': item['id'],
            'image_path': item['image_path'],
            'true_class': item['true_class'],
            'true_class_name': item['true_class_name'],
            'priority': item['priority']
        })
    
    conn.close()
    
    # Save to JSON file
    export_path = os.path.join(BASE_DIR, "src", "static", f'training_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(export_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    return jsonify({
        'success': True,
        'export_file': export_path,
        'total_samples': len(items)
    })

@app.route('/predict_dicom', methods=['POST'])
def predict_dicom():
    """Handle DICOM file predictions"""
    file = request.files['file']
    if file and file.filename.endswith('.dcm'):
        filename = secure_filename(file.filename)
        dicom_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(dicom_path)

        # --- Convert DICOM to PNG ---
        import pydicom, numpy as np
        from PIL import Image

        ds = pydicom.dcmread(dicom_path)
        img_array = ds.pixel_array.astype(float)
        img_array = (np.maximum(img_array, 0) / img_array.max()) * 255.0
        img_array = np.uint8(img_array)
        img = Image.fromarray(img_array)

        png_path = dicom_path.replace('.dcm', '.png')
        img.save(png_path)

        # --- Run ensemble prediction ---
        final_class, final_probs = ensemble_predict(
            png_path, yolo_model, resnet_model, vit_model, cnn_model,
            transform, device, weights
        )
        
        # Save to database
        prediction_id = save_prediction_to_db(
            filename, png_path, final_class, class_labels[final_class],
            final_probs[final_class] * 100, final_probs.tolist()
        )
        
        # Return all predictions
        all_predictions = []
        for i, prob in enumerate(final_probs):
            all_predictions.append({
                'class_name': class_labels[i],
                'class_id': i,
                'confidence': float(prob * 100)
            })
        
        # Sort by confidence
        all_predictions = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            "prediction_id": prediction_id,
            "predicted_class": class_labels[final_class],
            "confidence": float(final_probs[final_class] * 100),
            "all_predictions": all_predictions
        })
    else:
        return jsonify({"error": "Invalid file or format"}), 400

@app.route('/api/stats')
def api_stats():
    """API endpoint for getting current statistics"""
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    
    c.execute('SELECT COUNT(*) FROM predictions')
    total_predictions = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM feedback')
    total_feedback = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM feedback WHERE is_correct = 1')
    correct_predictions = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM training_queue WHERE used_in_training = 0')
    pending_training = c.fetchone()[0]
    
    accuracy = (correct_predictions / total_feedback * 100) if total_feedback > 0 else 0
    
    conn.close()
    
    return jsonify({
        'total_predictions': total_predictions,
        'total_feedback': total_feedback,
        'correct_predictions': correct_predictions,
        'accuracy': round(accuracy, 2),
        'pending_training_samples': pending_training
    })

# ---------------- Run ----------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
