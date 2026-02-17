"""
Ensemble Weight Optimization
Automatically optimize ensemble weights based on doctor feedback data
"""

import numpy as np
import sqlite3
import json
from datetime import datetime
from scipy.optimize import minimize
import torch
from PIL import Image
from torchvision import models, transforms
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class WeightOptimizer:
    def __init__(self, db_path=os.path.join(BASE_DIR, 'doctor_feedback.db'), models_dir=os.path.join(BASE_DIR, 'Ignore', 'Models')):
        self.db_path = db_path
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 5
        
        # Load models
        self.load_models()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def load_models(self):
        """Load all ensemble models"""
        print("📦 Loading models...")
        
        # YOLO
        self.yolo_model = YOLO(f"{self.models_dir}/best_yolo.pt")
        
        # ResNet
        self.resnet_model = models.resnet18()
        self.resnet_model.fc = torch.nn.Linear(self.resnet_model.fc.in_features, self.num_classes)
        self.resnet_model.load_state_dict(
            torch.load(f"{self.models_dir}/best_resnet.pth", map_location=self.device)
        )
        self.resnet_model.to(self.device).eval()
        
        # ViT
        self.vit_model = models.vit_b_16()
        self.vit_model.heads.head = torch.nn.Linear(
            self.vit_model.heads.head.in_features, self.num_classes
        )
        self.vit_model.load_state_dict(
            torch.load(f"{self.models_dir}/best_vit.pth", map_location=self.device)
        )
        self.vit_model.to(self.device).eval()
        
        # CNN
        self.cnn_model = load_model(f"{self.models_dir}/best_cnn.h5")
        
        print("✅ All models loaded successfully")
    
    def get_validated_predictions(self):
        """Get all predictions that have been validated by doctors"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''SELECT p.id, p.image_path, p.predicted_class, 
                            p.all_probabilities, f.is_correct, f.correct_class
                     FROM predictions p
                     JOIN feedback f ON p.id = f.prediction_id
                     WHERE f.is_correct IS NOT NULL''')
        
        data = []
        for row in c.fetchall():
            data.append({
                'id': row[0],
                'image_path': row[1],
                'predicted_class': row[2],
                'all_probs': json.loads(row[3]),
                'is_correct': row[4],
                'true_class': row[5] if not row[4] else row[2]
            })
        
        conn.close()
        
        print(f"📊 Retrieved {len(data)} validated predictions")
        return data
    
    def get_individual_predictions(self, image_path):
        """Get predictions from all individual models"""
        # YOLO
        yolo_preds = self.yolo_model.predict(image_path, verbose=False)
        yolo_probs = yolo_preds[0].probs.data.cpu().numpy()
        
        # ResNet
        img = Image.open(image_path).convert("RGB")
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            resnet_probs = torch.softmax(self.resnet_model(img_t), dim=1).cpu().numpy()[0]
        
        # ViT
        with torch.no_grad():
            vit_probs = torch.softmax(self.vit_model(img_t), dim=1).cpu().numpy()[0]
        
        # CNN
        img_cnn = keras_image.load_img(image_path, target_size=(150, 220))
        img_cnn = keras_image.img_to_array(img_cnn) / 255.0
        img_cnn = np.expand_dims(img_cnn, axis=0)
        cnn_probs = self.cnn_model.predict(img_cnn, verbose=0)[0]
        
        return {
            'yolo': yolo_probs,
            'resnet': resnet_probs,
            'vit': vit_probs,
            'cnn': cnn_probs
        }
    
    def objective_function(self, weights, predictions_data, true_labels):
        """
        Objective function to minimize (negative accuracy)
        weights: [w_yolo, w_resnet, w_vit, w_cnn]
        """
        w_yolo, w_resnet, w_vit, w_cnn = weights
        
        correct = 0
        total = len(predictions_data)
        
        for preds, true_label in zip(predictions_data, true_labels):
            # Weighted ensemble
            final_probs = (
                w_yolo * preds['yolo'] +
                w_resnet * preds['resnet'] +
                w_vit * preds['vit'] +
                w_cnn * preds['cnn']
            )
            
            predicted_class = np.argmax(final_probs)
            
            if predicted_class == true_label:
                correct += 1
        
        accuracy = correct / total
        return -accuracy  # Negative because we minimize
    
    def optimize_weights(self, validated_data):
        """Optimize ensemble weights using validated data"""
        print("\n🔧 Optimizing ensemble weights...")
        
        # Get individual model predictions for all validated samples
        print("📊 Collecting individual model predictions...")
        predictions_data = []
        true_labels = []
        
        for i, sample in enumerate(validated_data):
            if i % 10 == 0:
                print(f"   Processing {i}/{len(validated_data)}...", end='\r')
            
            preds = self.get_individual_predictions(sample['image_path'])
            predictions_data.append(preds)
            true_labels.append(sample['true_class'])
        
        print(f"\n✅ Collected predictions for {len(predictions_data)} samples")
        
        # Initial weights (current weights)
        initial_weights = np.array([0.3, 0.5, 0.05, 0.15])
        
        # Constraints: weights must sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: each weight between 0 and 1
        bounds = [(0, 1)] * 4
        
        # Optimize
        print("\n🎯 Running optimization...")
        result = minimize(
            self.objective_function,
            initial_weights,
            args=(predictions_data, true_labels),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'disp': True}
        )
        
        optimized_weights = result.x
        optimized_accuracy = -result.fun * 100
        
        # Calculate baseline accuracy with current weights
        baseline_accuracy = -self.objective_function(
            initial_weights, predictions_data, true_labels
        ) * 100
        
        print("\n" + "=" * 60)
        print("📊 OPTIMIZATION RESULTS")
        print("=" * 60)
        print(f"\n🔹 Baseline Accuracy (current weights): {baseline_accuracy:.2f}%")
        print(f"🔹 Optimized Accuracy: {optimized_accuracy:.2f}%")
        print(f"🔹 Improvement: {optimized_accuracy - baseline_accuracy:.2f}%")
        
        print("\n📊 Weight Comparison:")
        print(f"{'Model':<10} {'Current':<12} {'Optimized':<12} {'Change':<12}")
        print("-" * 50)
        
        model_names = ['YOLO', 'ResNet', 'ViT', 'CNN']
        for i, name in enumerate(model_names):
            change = optimized_weights[i] - initial_weights[i]
            change_str = f"+{change:.3f}" if change >= 0 else f"{change:.3f}"
            print(f"{name:<10} {initial_weights[i]:<12.3f} {optimized_weights[i]:<12.3f} {change_str:<12}")
        
        return {
            'optimized_weights': {
                'yolo': float(optimized_weights[0]),
                'resnet': float(optimized_weights[1]),
                'vit': float(optimized_weights[2]),
                'cnn': float(optimized_weights[3])
            },
            'baseline_accuracy': baseline_accuracy,
            'optimized_accuracy': optimized_accuracy,
            'improvement': optimized_accuracy - baseline_accuracy,
            'samples_used': len(validated_data),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_optimized_weights(self, results):
        """Save optimized weights to Python file"""
        output_file = 'ensemble_optimized.py'
        
        with open(output_file, 'w') as f:
            f.write(f"""# Optimized Ensemble Weights
# Generated: {results['timestamp']}
# Optimization Results:
#   - Samples used: {results['samples_used']}
#   - Baseline accuracy: {results['baseline_accuracy']:.2f}%
#   - Optimized accuracy: {results['optimized_accuracy']:.2f}%
#   - Improvement: {results['improvement']:.2f}%

weights = {{
    "yolo": {results['optimized_weights']['yolo']:.4f},
    "resnet": {results['optimized_weights']['resnet']:.4f},
    "vit": {results['optimized_weights']['vit']:.4f},
    "cnn": {results['optimized_weights']['cnn']:.4f}
}}
""")
        
        print(f"\n✅ Optimized weights saved to: {output_file}")
        print("\n💡 To use optimized weights:")
        print(f"   1. Review the results above")
        print(f"   2. If improvement is significant, replace ensemble.py weights")
        print(f"   3. Test on validation set before deployment")
        
        return output_file
    
    def run_optimization(self):
        """Execute full weight optimization pipeline"""
        print("=" * 60)
        print("🚀 ENSEMBLE WEIGHT OPTIMIZATION PIPELINE")
        print("=" * 60)
        
        # Get validated data
        validated_data = self.get_validated_predictions()
        
        if len(validated_data) < 10:
            print(f"\n⚠️  Insufficient validated samples (found {len(validated_data)}, need ≥10)")
            return None
        
        # Optimize weights
        results = self.optimize_weights(validated_data)
        
        # Save results
        output_file = self.save_optimized_weights(results)
        
        # Save JSON report
        json_file = f'weight_optimization_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Detailed report saved to: {json_file}")
        
        return results


def main():
    """Main execution"""
    optimizer = WeightOptimizer()
    results = optimizer.run_optimization()
    
    if results:
        print("\n✅ Weight optimization completed successfully!")
        
        if results['improvement'] > 1.0:
            print("\n🎉 Significant improvement detected!")
            print("   Consider updating ensemble.py with the new weights.")
        elif results['improvement'] > 0:
            print("\n✅ Minor improvement detected.")
            print("   Review carefully before updating weights.")
        else:
            print("\n⚠️  No improvement detected.")
            print("   Current weights are optimal. No changes needed.")
    else:
        print("\n⚠️  Optimization failed - insufficient data")


if __name__ == "__main__":
    main()
