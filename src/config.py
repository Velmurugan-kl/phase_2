"""
Configuration Management
Centralized configuration for the Doctor-in-the-Loop system
"""

import os
from dataclasses import dataclass
from typing import Dict, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class ModelConfig:
    """Model-specific configuration"""
    yolo_path: str = os.path.join(BASE_DIR, "Ignore", "Models", "best_yolo.pt")
    resnet_path: str = os.path.join(BASE_DIR, "Ignore", "Models", "best_resnet.pth")
    vit_path: str = os.path.join(BASE_DIR, "Ignore", "Models", "best_vit.pth")
    cnn_path: str = os.path.join(BASE_DIR, "Ignore", "Models", "best_cnn.h5")
    retrained_dir: str = os.path.join(BASE_DIR, "Ignore", "Models", "retrained")


@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_path: str = os.path.join(BASE_DIR, "doctor_feedback.db")
    backup_dir: str = os.path.join(BASE_DIR, "backups")
    auto_backup: bool = True
    backup_interval_days: int = 7


@dataclass
class TrainingConfig:
    """Training and retraining configuration"""
    min_samples_for_retraining: int = 10
    recommended_samples: int = 50
    optimal_samples: int = 100
    default_epochs: int = 10
    default_batch_size: int = 8
    learning_rate: float = 0.0001
    validation_split: float = 0.2
    early_stopping_patience: int = 3


@dataclass
class EnsembleConfig:
    """Ensemble model configuration"""
    weights: Dict[str, float] = None
    num_classes: int = 5
    class_labels: List[str] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "yolo": 0.3,
                "resnet": 0.5,
                "vit": 0.05,
                "cnn": 0.15
            }
        
        if self.class_labels is None:
            self.class_labels = [
                "Bacterial Pneumonia",
                "Corona Virus Disease",
                "Normal",
                "Tuberculosis",
                "Viral Pneumonia"
            ]


@dataclass
class WebConfig:
    """Web application configuration"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = True
    secret_key: str = "secret key"  # Change in production!
    upload_folder: str = "static/uploads"
    feedback_folder: str = "static/feedback_images"
    max_upload_size_mb: int = 16
    allowed_extensions: set = None
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = {'png', 'jpg', 'jpeg', 'dcm'}


@dataclass
class OptimizationConfig:
    """Weight optimization configuration"""
    min_samples_for_optimization: int = 20
    max_iterations: int = 100
    improvement_threshold: float = 0.5  # Minimum improvement % to update weights
    significant_improvement: float = 1.0  # Improvement % considered significant


@dataclass
class PerformanceConfig:
    """Performance tracking configuration"""
    target_accuracy: float = 85.0
    good_accuracy: float = 90.0
    excellent_accuracy: float = 95.0
    min_precision: float = 80.0
    min_recall: float = 80.0
    min_f1_score: float = 80.0


@dataclass
class DataCleaningConfig:
    """Data cleaning configuration"""
    auto_clean_enabled: bool = False
    clean_after_days: int = 90
    keep_feedback_samples: bool = True


class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.database = DatabaseConfig()
        self.training = TrainingConfig()
        self.ensemble = EnsembleConfig()
        self.web = WebConfig()
        self.optimization = OptimizationConfig()
        self.performance = PerformanceConfig()
        self.data_cleaning = DataCleaningConfig()
    
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.web.upload_folder,
            self.web.feedback_folder,
            self.model.retrained_dir,
            self.database.backup_dir,
            "visualizations",
            "static"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def validate(self):
        """Validate configuration"""
        errors = []
        
        # Check model files exist
        if not os.path.exists(self.model.yolo_path):
            errors.append(f"YOLO model not found: {self.model.yolo_path}")
        if not os.path.exists(self.model.resnet_path):
            errors.append(f"ResNet model not found: {self.model.resnet_path}")
        if not os.path.exists(self.model.vit_path):
            errors.append(f"ViT model not found: {self.model.vit_path}")
        if not os.path.exists(self.model.cnn_path):
            errors.append(f"CNN model not found: {self.model.cnn_path}")
        
        # Check weights sum to 1
        weight_sum = sum(self.ensemble.weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            errors.append(f"Ensemble weights must sum to 1.0, got {weight_sum}")
        
        # Check number of classes matches labels
        if len(self.ensemble.class_labels) != self.ensemble.num_classes:
            errors.append(f"Number of classes mismatch: {len(self.ensemble.class_labels)} labels vs {self.ensemble.num_classes} classes")
        
        if errors:
            print("⚠️  Configuration Validation Errors:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        print("✅ Configuration validated successfully")
        return True
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "=" * 60)
        print("⚙️  SYSTEM CONFIGURATION")
        print("=" * 60)
        
        print("\n📦 Models:")
        print(f"   YOLO:   {self.model.yolo_path}")
        print(f"   ResNet: {self.model.resnet_path}")
        print(f"   ViT:    {self.model.vit_path}")
        print(f"   CNN:    {self.model.cnn_path}")
        
        print("\n🗄️  Database:")
        print(f"   Path: {self.database.db_path}")
        print(f"   Auto-backup: {self.database.auto_backup}")
        
        print("\n🎯 Ensemble Weights:")
        for model, weight in self.ensemble.weights.items():
            print(f"   {model.upper()}: {weight:.3f}")
        
        print("\n🔄 Training:")
        print(f"   Min samples: {self.training.min_samples_for_retraining}")
        print(f"   Default epochs: {self.training.default_epochs}")
        print(f"   Batch size: {self.training.default_batch_size}")
        
        print("\n🌐 Web Server:")
        print(f"   Host: {self.web.host}")
        print(f"   Port: {self.web.port}")
        print(f"   Debug: {self.web.debug}")
        
        print("\n📊 Performance Targets:")
        print(f"   Target accuracy: {self.performance.target_accuracy}%")
        print(f"   Good accuracy: {self.performance.good_accuracy}%")
        print(f"   Excellent accuracy: {self.performance.excellent_accuracy}%")
        
        print("=" * 60 + "\n")
    
    def save_to_file(self, filepath='config.json'):
        """Save configuration to JSON file"""
        import json
        
        config_dict = {
            'model': {
                'yolo_path': self.model.yolo_path,
                'resnet_path': self.model.resnet_path,
                'vit_path': self.model.vit_path,
                'cnn_path': self.model.cnn_path,
                'retrained_dir': self.model.retrained_dir
            },
            'database': {
                'db_path': self.database.db_path,
                'backup_dir': self.database.backup_dir,
                'auto_backup': self.database.auto_backup
            },
            'training': {
                'min_samples_for_retraining': self.training.min_samples_for_retraining,
                'default_epochs': self.training.default_epochs,
                'default_batch_size': self.training.default_batch_size,
                'learning_rate': self.training.learning_rate
            },
            'ensemble': {
                'weights': self.ensemble.weights,
                'num_classes': self.ensemble.num_classes,
                'class_labels': self.ensemble.class_labels
            },
            'web': {
                'host': self.web.host,
                'port': self.web.port,
                'debug': self.web.debug,
                'upload_folder': self.web.upload_folder
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✅ Configuration saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath='config.json'):
        """Load configuration from JSON file"""
        import json
        
        config = cls()
        
        if not os.path.exists(filepath):
            print(f"⚠️  Config file not found: {filepath}")
            print("   Using default configuration")
            return config
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Update configuration
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                setattr(config.model, key, value)
        
        if 'database' in config_dict:
            for key, value in config_dict['database'].items():
                setattr(config.database, key, value)
        
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                setattr(config.training, key, value)
        
        if 'ensemble' in config_dict:
            for key, value in config_dict['ensemble'].items():
                setattr(config.ensemble, key, value)
        
        if 'web' in config_dict:
            for key, value in config_dict['web'].items():
                if key != 'allowed_extensions':  # Skip set conversion
                    setattr(config.web, key, value)
        
        print(f"✅ Configuration loaded from {filepath}")
        return config


# Global configuration instance
config = Config()


def main():
    """Configuration management CLI"""
    print("⚙️  Configuration Management Tool")
    print("=" * 60)
    
    config.print_config()
    
    print("\nOptions:")
    print("1. Validate configuration")
    print("2. Create directories")
    print("3. Save configuration to file")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        config.validate()
    elif choice == '2':
        config.create_directories()
        print("✅ Directories created")
    elif choice == '3':
        filename = input("Enter filename (default: config.json): ").strip()
        if not filename:
            filename = 'config.json'
        config.save_to_file(filename)
    elif choice == '4':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
