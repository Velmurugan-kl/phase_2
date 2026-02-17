"""
Automated Model Retraining Pipeline
This script handles the complete retraining workflow using doctor-corrected samples
"""

import os
import json
import sqlite3
import shutil
from datetime import datetime
import numpy as np
from pathlib import Path

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# TensorFlow imports
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# YOLO imports
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class FeedbackDataset(Dataset):
    """Custom PyTorch Dataset for feedback samples"""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        label = sample['true_class']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class RetrainingPipeline:
    def __init__(self, db_path=os.path.join(BASE_DIR, 'doctor_feedback.db'), models_dir=os.path.join(BASE_DIR, 'Ignore', 'Models'), 
                 retrained_dir=os.path.join(BASE_DIR, 'Ignore', 'Models', 'retrained')):
        self.db_path = db_path
        self.models_dir = models_dir
        self.retrained_dir = retrained_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create retrained models directory
        os.makedirs(retrained_dir, exist_ok=True)
        
        # Training parameters
        self.num_classes = 5
        self.class_labels = [
            "Bacterial Pneumonia", 
            "Corona Virus Disease", 
            "Normal", 
            "Tuberculosis", 
            "Viral Pneumonia"
        ]
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def get_feedback_samples(self):
        """Retrieve all unused feedback samples from database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''SELECT id, image_path, true_class, true_class_name, priority
                     FROM training_queue
                     WHERE used_in_training = 0
                     ORDER BY priority DESC, added_timestamp ASC''')
        
        samples = []
        for row in c.fetchall():
            samples.append({
                'id': row[0],
                'image_path': row[1],
                'true_class': row[2],
                'true_class_name': row[3],
                'priority': row[4]
            })
        
        conn.close()
        
        print(f"📊 Retrieved {len(samples)} feedback samples for retraining")
        return samples
    
    def prepare_datasets(self, samples, validation_split=0.2):
        """Split samples into training and validation sets"""
        np.random.shuffle(samples)
        split_idx = int(len(samples) * (1 - validation_split))
        
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        train_dataset = FeedbackDataset(train_samples, self.transform)
        val_dataset = FeedbackDataset(val_samples, self.val_transform)
        
        return train_dataset, val_dataset
    
    def retrain_resnet(self, train_loader, val_loader, epochs=10):
        """Fine-tune ResNet18 model with feedback data"""
        print("\n🔄 Retraining ResNet18...")
        
        # Load existing model
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        model.load_state_dict(torch.load(
            os.path.join(self.models_dir, 'best_resnet.pth'),
            map_location=self.device
        ))
        model.to(self.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR for fine-tuning
        
        best_acc = 0.0
        save_path = os.path.join(self.retrained_dir, 
                                f'resnet_retrained_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Validation phase
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_acc = 100 * correct / total
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_path)
                print(f"✅ Best model saved with Val Acc: {best_acc:.2f}%")
        
        return save_path, best_acc
    
    def retrain_vit(self, train_loader, val_loader, epochs=10):
        """Fine-tune Vision Transformer model with feedback data"""
        print("\n🔄 Retraining ViT...")
        
        # Load existing model
        model = models.vit_b_16()
        model.heads.head = nn.Linear(model.heads.head.in_features, self.num_classes)
        model.load_state_dict(torch.load(
            os.path.join(self.models_dir, 'best_vit.pth'),
            map_location=self.device
        ))
        model.to(self.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        best_acc = 0.0
        save_path = os.path.join(self.retrained_dir, 
                                f'vit_retrained_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_acc = 100 * correct / total
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_path)
                print(f"✅ Best model saved with Val Acc: {best_acc:.2f}%")
        
        return save_path, best_acc
    
    def retrain_cnn(self, samples, epochs=10):
        """Fine-tune CNN model with feedback data"""
        print("\n🔄 Retraining CNN...")
        
        # Prepare data for Keras
        X_train = []
        y_train = []
        
        for sample in samples:
            img = keras_image.load_img(sample['image_path'], target_size=(150, 220))
            img_array = keras_image.img_to_array(img) / 255.0
            X_train.append(img_array)
            y_train.append(sample['true_class'])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Convert to categorical
        from tensorflow.keras.utils import to_categorical
        y_train = to_categorical(y_train, num_classes=self.num_classes)
        
        # Load existing model
        model = load_model(os.path.join(self.models_dir, 'best_cnn.h5'))
        
        # Compile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        save_path = os.path.join(self.retrained_dir, 
                                f'cnn_retrained_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
        
        # Callbacks
        checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', 
                                     save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1)
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=16,
            callbacks=[checkpoint, early_stop],
            verbose=1
        )
        
        best_acc = max(history.history['val_accuracy']) * 100
        print(f"✅ Best CNN Val Acc: {best_acc:.2f}%")
        
        return save_path, best_acc
    
    def retrain_yolo(self, samples):
        """Retrain YOLO classifier with feedback data"""
        print("\n🔄 Retraining YOLOv8 Classifier...")
        print("⚠️  YOLO retraining requires organized dataset structure.")
        print("    Creating temporary dataset structure...")
        
        # Create temporary dataset structure for YOLO
        temp_dataset = 'temp_yolo_feedback_dataset'
        os.makedirs(temp_dataset, exist_ok=True)
        
        for class_name in self.class_labels:
            os.makedirs(os.path.join(temp_dataset, class_name), exist_ok=True)
        
        # Copy images to appropriate class folders
        for sample in samples:
            src = sample['image_path']
            class_folder = self.class_labels[sample['true_class']]
            dst = os.path.join(temp_dataset, class_folder, os.path.basename(src))
            shutil.copy2(src, dst)
        
        # Load base model and retrain
        model = YOLO(os.path.join(self.models_dir, 'best_yolo.pt'))
        
        save_path = os.path.join(self.retrained_dir, 
                                f'yolo_retrained_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
        
        results = model.train(
            data=temp_dataset,
            epochs=10,
            imgsz=224,
            batch=8,
            device=0 if torch.cuda.is_available() else 'cpu',
            project=self.retrained_dir,
            name='yolo_retrain'
        )
        
        # Clean up temp dataset
        shutil.rmtree(temp_dataset)
        
        print("✅ YOLO retraining completed!")
        return results
    
    def mark_samples_as_used(self, sample_ids):
        """Mark samples as used in training"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for sample_id in sample_ids:
            c.execute('''UPDATE training_queue 
                        SET used_in_training = 1 
                        WHERE id = ?''', (sample_id,))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Marked {len(sample_ids)} samples as used in training")
    
    def save_retraining_report(self, results):
        """Save retraining results to JSON report"""
        report_path = os.path.join(
            self.retrained_dir,
            f'retraining_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Retraining report saved to: {report_path}")
        return report_path
    
    def run_full_pipeline(self, min_samples=10, epochs=10, batch_size=8):
        """Execute complete retraining pipeline"""
        print("=" * 80)
        print("🚀 STARTING AUTOMATED RETRAINING PIPELINE")
        print("=" * 80)
        
        # Step 1: Get feedback samples
        samples = self.get_feedback_samples()
        
        if len(samples) < min_samples:
            print(f"⚠️  Insufficient samples for retraining (found {len(samples)}, need {min_samples})")
            return None
        
        print(f"✅ Found {len(samples)} samples - proceeding with retraining")
        
        # Step 2: Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets(samples)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Step 3: Retrain models
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(samples),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'epochs': epochs,
            'models': {}
        }
        
        # Retrain ResNet
        try:
            resnet_path, resnet_acc = self.retrain_resnet(train_loader, val_loader, epochs)
            results['models']['resnet'] = {
                'path': resnet_path,
                'val_accuracy': resnet_acc
            }
        except Exception as e:
            print(f"❌ ResNet retraining failed: {str(e)}")
            results['models']['resnet'] = {'error': str(e)}
        
        # Retrain ViT
        try:
            vit_path, vit_acc = self.retrain_vit(train_loader, val_loader, epochs)
            results['models']['vit'] = {
                'path': vit_path,
                'val_accuracy': vit_acc
            }
        except Exception as e:
            print(f"❌ ViT retraining failed: {str(e)}")
            results['models']['vit'] = {'error': str(e)}
        
        # Retrain CNN
        try:
            cnn_path, cnn_acc = self.retrain_cnn(samples, epochs)
            results['models']['cnn'] = {
                'path': cnn_path,
                'val_accuracy': cnn_acc
            }
        except Exception as e:
            print(f"❌ CNN retraining failed: {str(e)}")
            results['models']['cnn'] = {'error': str(e)}
        
        # Step 4: Mark samples as used
        sample_ids = [s['id'] for s in samples]
        self.mark_samples_as_used(sample_ids)
        
        # Step 5: Save report
        report_path = self.save_retraining_report(results)
        
        print("\n" + "=" * 80)
        print("✅ RETRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\n📊 Summary:")
        print(f"   - Total samples processed: {len(samples)}")
        print(f"   - Models retrained: {len([m for m in results['models'] if 'error' not in results['models'][m]])}")
        print(f"   - Report saved to: {report_path}")
        print("\n💡 Next steps:")
        print("   1. Review the retraining report")
        print("   2. Test retrained models on validation set")
        print("   3. Update ensemble weights if needed")
        print("   4. Deploy new models to production")
        
        return results


def main():
    """Main execution function"""
    pipeline = RetrainingPipeline()
    
    # Run full retraining pipeline
    results = pipeline.run_full_pipeline(
        min_samples=5,  # Minimum 5 samples to start retraining
        epochs=10,
        batch_size=8
    )
    
    if results:
        print("\n✅ Retraining completed successfully!")
    else:
        print("\n⚠️  Retraining skipped - insufficient samples")


if __name__ == "__main__":
    main()
