#!/usr/bin/env python3
"""
Poetry-Based Setup Script for Doctor-in-the-Loop System
Automated setup using Poetry for dependency management
"""

import os
import sys
import subprocess
import sqlite3
from pathlib import Path


class PoetrySetupWizard:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.errors = []
        self.warnings = []
    
    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "=" * 70)
        print(f"  {text}")
        print("=" * 70 + "\n")
    
    def print_step(self, step_num, total_steps, description):
        """Print step progress"""
        print(f"\n[{step_num}/{total_steps}] {description}")
        print("-" * 70)
    
    def check_python_version(self):
        """Check Python version"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.errors.append("Python 3.8+ is required")
            return False
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_poetry(self):
        """Check if Poetry is installed"""
        try:
            result = subprocess.run(['poetry', '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  check=True)
            print(f"✅ Poetry is installed: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Poetry is not installed")
            print("\n📦 Installing Poetry...")
            return self.install_poetry()
    
    def install_poetry(self):
        """Install Poetry"""
        try:
            print("   Downloading and installing Poetry...")
            
            # Try official installation method
            if sys.platform == 'win32':
                install_cmd = '(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -'
                subprocess.run(['powershell', '-Command', install_cmd], check=True)
            else:
                install_cmd = 'curl -sSL https://install.python-poetry.org | python3 -'
                subprocess.run(install_cmd, shell=True, check=True)
            
            print("✅ Poetry installed successfully")
            print("⚠️  You may need to restart your terminal/shell")
            print("   Then run this script again")
            return True
        except Exception as e:
            self.errors.append(f"Failed to install Poetry: {e}")
            print(f"\n❌ Automatic installation failed: {e}")
            print("\n💡 Please install Poetry manually:")
            print("   Visit: https://python-poetry.org/docs/#installation")
            print("\n   Quick install:")
            if sys.platform == 'win32':
                print("   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -")
            else:
                print("   curl -sSL https://install.python-poetry.org | python3 -")
            return False
    
    def create_directory_structure(self):
        """Create necessary directories"""
        print("\n📁 Creating directory structure...")
        
        directories = [
            'Models',
            'Models/retrained',
            'static/uploads',
            'static/feedback_images',
            'templates',
            'backups',
            'visualizations',
            'logs'
        ]
        
        for directory in directories:
            dir_path = self.base_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}")
        
        return True
    
    def check_pyproject_toml(self):
        """Check if pyproject.toml exists"""
        pyproject_path = self.base_dir / 'pyproject.toml'
        if not pyproject_path.exists():
            print("⚠️  pyproject.toml not found")
            self.warnings.append("pyproject.toml is missing")
            return False
        print("✅ pyproject.toml found")
        return True
    
    def install_dependencies(self):
        """Install dependencies using Poetry"""
        print("\n📦 Installing dependencies with Poetry...")
        print("   This may take several minutes...")
        
        try:
            # Install dependencies
            subprocess.run(['poetry', 'install'], check=True)
            print("✅ All dependencies installed successfully")
            
            # Show virtual environment info
            result = subprocess.run(['poetry', 'env', 'info'], 
                                  capture_output=True, 
                                  text=True)
            print("\n📍 Virtual Environment Info:")
            print(result.stdout)
            
            return True
        except subprocess.CalledProcessError as e:
            self.errors.append(f"Failed to install dependencies: {e}")
            return False
    
    def check_models(self):
        """Check if model files exist"""
        print("\n🔍 Checking for model files...")
        # Support models placed either in 'Models/' or 'Ignore/Models/'
        candidate_dirs = [self.base_dir / 'Models', self.base_dir / 'Ignore' / 'Models']
        models = {
            'YOLO': 'best_yolo.pt',
            'ResNet': 'best_resnet.pth',
            'ViT': 'best_vit.pth',
            'CNN': 'best_cnn.h5'
        }

        all_exist = True
        for name, fname in models.items():
            found = False
            found_path = None
            for d in candidate_dirs:
                model_path = d / fname
                if model_path.exists():
                    found = True
                    found_path = model_path
                    break
            if found:
                print(f"✅ {name} model found: {found_path}")
            else:
                print(f"⚠️  {name} model NOT found in Models/ or Ignore/Models/")
                self.warnings.append(f"Missing {name} model (looked in Models/ and Ignore/Models/)")
                all_exist = False
        
        if not all_exist:
            print("\n💡 Note: You'll need to add model files before running the application")
            print("   Place your trained models in Models/ or Ignore/Models/")
        
        return all_exist
    
    def initialize_database(self):
        """Initialize SQLite database"""
        print("\n🗄️  Initializing database...")
        
        db_path = self.base_dir / 'doctor_feedback.db'
        
        try:
            conn = sqlite3.connect(str(db_path))
            c = conn.cursor()
            
            # Create tables
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
            
            c.execute('''CREATE TABLE IF NOT EXISTS model_performance
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          model_version TEXT NOT NULL,
                          total_predictions INTEGER DEFAULT 0,
                          correct_predictions INTEGER DEFAULT 0,
                          accuracy REAL DEFAULT 0.0,
                          last_updated DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            
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
            
            print(f"✅ Database initialized: {db_path}")
            return True
        
        except Exception as e:
            self.errors.append(f"Database initialization failed: {e}")
            return False
    
    def create_config_file(self):
        """Create default configuration file"""
        print("\n⚙️  Creating configuration file...")
        
        config_content = """{
  "model": {
    "yolo_path": "Models/best_yolo.pt",
    "resnet_path": "Models/best_resnet.pth",
    "vit_path": "Models/best_vit.pth",
    "cnn_path": "Models/best_cnn.h5",
    "retrained_dir": "Models/retrained"
  },
  "database": {
    "db_path": "doctor_feedback.db",
    "backup_dir": "backups",
    "auto_backup": true
  },
  "training": {
    "min_samples_for_retraining": 10,
    "default_epochs": 10,
    "default_batch_size": 8,
    "learning_rate": 0.0001
  },
  "ensemble": {
    "weights": {
      "yolo": 0.3,
      "resnet": 0.5,
      "vit": 0.05,
      "cnn": 0.15
    },
    "num_classes": 5,
    "class_labels": [
      "Bacterial Pneumonia",
      "Corona Virus Disease",
      "Normal",
      "Tuberculosis",
      "Viral Pneumonia"
    ]
  },
  "web": {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": true
  }
}
"""
        
        with open('config.json', 'w') as f:
            f.write(config_content)
        
        print("✅ Configuration file created: config.json")
        return True
    
    def create_gitignore(self):
        """Create .gitignore file"""
        print("\n📝 Creating .gitignore...")
        
        gitignore_content = """# Database
*.db
*.db-journal

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# Poetry
poetry.lock

# Models (too large for git)
Models/*.pt
Models/*.pth
Models/*.h5
Models/retrained/

# Generated files
static/uploads/*
static/feedback_images/*
backups/*
visualizations/*
logs/*

# Config (may contain secrets)
config.json

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Coverage
.coverage
htmlcov/
"""
        
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        
        print("✅ .gitignore created")
        return True
    
    def verify_installation(self):
        """Verify installation"""
        print("\n🔍 Verifying installation...")
        
        checks = {
            'Database': (self.base_dir / 'doctor_feedback.db').exists(),
            'Config': (self.base_dir / 'config.json').exists(),
            'pyproject.toml': (self.base_dir / 'pyproject.toml').exists(),
            'Upload folder': (self.base_dir / 'static' / 'uploads').exists(),
            'Templates': (self.base_dir / 'templates').exists(),
            '.gitignore': (self.base_dir / '.gitignore').exists(),
        }
        
        all_passed = True
        for check_name, passed in checks.items():
            if passed:
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}")
                all_passed = False
        
        return all_passed
    
    def print_summary(self):
        """Print setup summary"""
        self.print_header("SETUP SUMMARY")
        
        if self.errors:
            print("❌ ERRORS:")
            for error in self.errors:
                print(f"   - {error}")
            print()
        
        if self.warnings:
            print("⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   - {warning}")
            print()
        
        if not self.errors:
            print("✅ Setup completed successfully!")
            print("\n📋 Next Steps:")
            print("   1. Place your trained models in Models/ directory:")
            print("      - best_yolo.pt")
            print("      - best_resnet.pth")
            print("      - best_vit.pth")
            print("      - best_cnn.h5")
            print("\n   2. Copy HTML templates to templates/ directory:")
            print("      - index.html")
            print("      - result_enhanced.html")
            print("      - dashboard.html")
            print("      - training_queue.html")
            print("\n   3. Activate Poetry virtual environment:")
            print("      poetry shell")
            print("\n   4. Start the application:")
            print("      poetry run python app_enhanced.py")
            print("      # OR use the CLI command:")
            print("      poetry run lung-app")
            print("\n   5. Open browser:")
            print("      http://localhost:5000")
            print("\n💡 Useful Poetry Commands:")
            print("   poetry shell                  # Activate virtual environment")
            print("   poetry install                # Install dependencies")
            print("   poetry add <package>          # Add new dependency")
            print("   poetry update                 # Update dependencies")
            print("   poetry run lung-app           # Run main app")
            print("   poetry run lung-retrain       # Run retraining")
            print("   poetry run lung-optimize      # Optimize weights")
            print("   poetry run lung-db            # Database management")
        else:
            print("❌ Setup failed. Please fix errors and try again.")
    
    def run(self):
        """Run complete setup"""
        self.print_header("DOCTOR-IN-THE-LOOP SYSTEM - POETRY SETUP")
        
        total_steps = 9
        
        # Step 1: Check Python version
        self.print_step(1, total_steps, "Checking Python version")
        if not self.check_python_version():
            self.print_summary()
            return False
        
        # Step 2: Check Poetry
        self.print_step(2, total_steps, "Checking Poetry installation")
        if not self.check_poetry():
            self.print_summary()
            return False
        
        # Step 3: Check pyproject.toml
        self.print_step(3, total_steps, "Checking pyproject.toml")
        self.check_pyproject_toml()
        
        # Step 4: Create directories
        self.print_step(4, total_steps, "Creating directory structure")
        self.create_directory_structure()
        
        # Step 5: Install dependencies
        self.print_step(5, total_steps, "Installing dependencies with Poetry")
        install_deps = input("Install dependencies now? (y/n): ").lower().strip()
        if install_deps == 'y':
            self.install_dependencies()
        else:
            print("⏭️  Skipping dependency installation")
            print("   Run later with: poetry install")
            self.warnings.append("Dependencies not installed. Run: poetry install")
        
        # Step 6: Check models
        self.print_step(6, total_steps, "Checking model files")
        self.check_models()
        
        # Step 7: Initialize database
        self.print_step(7, total_steps, "Initializing database")
        self.initialize_database()
        
        # Step 8: Create config and gitignore
        self.print_step(8, total_steps, "Creating configuration files")
        self.create_config_file()
        self.create_gitignore()
        
        # Step 9: Verify
        self.print_step(9, total_steps, "Verifying installation")
        self.verify_installation()
        
        # Summary
        self.print_summary()
        
        return len(self.errors) == 0


def main():
    """Main setup function"""
    print("\n" + "🚀 " * 25)
    print("   DOCTOR-IN-THE-LOOP LEARNING FRAMEWORK - POETRY SETUP")
    print("🚀 " * 25 + "\n")
    
    print("This wizard will set up the complete system using Poetry:")
    print("  ✓ Poetry dependency management")
    print("  ✓ Virtual environment")
    print("  ✓ Directory structure")
    print("  ✓ Database initialization")
    print("  ✓ Configuration files")
    print()
    
    proceed = input("Continue with setup? (y/n): ").lower().strip()
    
    if proceed != 'y':
        print("\n👋 Setup cancelled.")
        return
    
    wizard = PoetrySetupWizard()
    success = wizard.run()
    
    if success:
        print("\n" + "🎉 " * 25)
        print("   SETUP COMPLETE!")
        print("🎉 " * 25)
    else:
        print("\n" + "❌ " * 25)
        print("   SETUP FAILED - Please review errors above")
        print("❌ " * 25)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
