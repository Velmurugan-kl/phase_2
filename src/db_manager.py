"""
Database Management Utilities
Tools for managing, analyzing, and maintaining the feedback database
"""

import sqlite3
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DatabaseManager:
    def __init__(self, db_path=os.path.join(BASE_DIR, 'doctor_feedback.db')):
        self.db_path = db_path
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def get_statistics(self):
        """Get comprehensive database statistics"""
        conn = self.get_connection()
        c = conn.cursor()
        
        stats = {}
        
        # Total predictions
        c.execute('SELECT COUNT(*) FROM predictions')
        stats['total_predictions'] = c.fetchone()[0]
        
        # Total feedback
        c.execute('SELECT COUNT(*) FROM feedback')
        stats['total_feedback'] = c.fetchone()[0]
        
        # Correct vs incorrect
        c.execute('SELECT COUNT(*) FROM feedback WHERE is_correct = 1')
        stats['correct_predictions'] = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM feedback WHERE is_correct = 0')
        stats['incorrect_predictions'] = c.fetchone()[0]
        
        # Accuracy
        if stats['total_feedback'] > 0:
            stats['accuracy'] = (stats['correct_predictions'] / stats['total_feedback']) * 100
        else:
            stats['accuracy'] = 0
        
        # Training queue
        c.execute('SELECT COUNT(*) FROM training_queue WHERE used_in_training = 0')
        stats['pending_training'] = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM training_queue WHERE used_in_training = 1')
        stats['used_in_training'] = c.fetchone()[0]
        
        # Class distribution
        c.execute('''SELECT predicted_class_name, COUNT(*) 
                     FROM predictions 
                     GROUP BY predicted_class_name''')
        stats['class_distribution'] = dict(c.fetchall())
        
        # Feedback distribution by class
        c.execute('''SELECT p.predicted_class_name, 
                            SUM(CASE WHEN f.is_correct = 1 THEN 1 ELSE 0 END) as correct,
                            SUM(CASE WHEN f.is_correct = 0 THEN 1 ELSE 0 END) as incorrect
                     FROM predictions p
                     JOIN feedback f ON p.id = f.prediction_id
                     GROUP BY p.predicted_class_name''')
        
        stats['feedback_by_class'] = {}
        for row in c.fetchall():
            stats['feedback_by_class'][row[0]] = {
                'correct': row[1],
                'incorrect': row[2]
            }
        
        conn.close()
        return stats
    
    def export_to_excel(self, output_file='feedback_report.xlsx'):
        """Export all data to Excel for analysis"""
        conn = self.get_connection()
        
        # Get all tables
        predictions_df = pd.read_sql_query("SELECT * FROM predictions", conn)
        feedback_df = pd.read_sql_query("SELECT * FROM feedback", conn)
        training_queue_df = pd.read_sql_query("SELECT * FROM training_queue", conn)
        performance_df = pd.read_sql_query("SELECT * FROM model_performance", conn)
        class_perf_df = pd.read_sql_query("SELECT * FROM class_performance", conn)
        
        # Create Excel writer
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
            feedback_df.to_excel(writer, sheet_name='Feedback', index=False)
            training_queue_df.to_excel(writer, sheet_name='Training Queue', index=False)
            performance_df.to_excel(writer, sheet_name='Overall Performance', index=False)
            class_perf_df.to_excel(writer, sheet_name='Class Performance', index=False)
        
        conn.close()
        print(f"✅ Data exported to {output_file}")
        return output_file
    
    def generate_visualizations(self, output_dir='visualizations'):
        """Generate visualization plots"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        conn = self.get_connection()
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Accuracy over time
        df = pd.read_sql_query('''
            SELECT DATE(f.feedback_timestamp) as date,
                   AVG(CASE WHEN f.is_correct = 1 THEN 1 ELSE 0 END) * 100 as accuracy
            FROM feedback f
            GROUP BY DATE(f.feedback_timestamp)
            ORDER BY date
        ''', conn)
        
        if not df.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(df['date'], df['accuracy'], marker='o', linewidth=2)
            plt.xlabel('Date')
            plt.ylabel('Accuracy (%)')
            plt.title('Model Accuracy Over Time')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/accuracy_over_time.png', dpi=300)
            plt.close()
        
        # 2. Class distribution
        df = pd.read_sql_query('''
            SELECT predicted_class_name, COUNT(*) as count
            FROM predictions
            GROUP BY predicted_class_name
        ''', conn)
        
        plt.figure(figsize=(10, 6))
        plt.bar(df['predicted_class_name'], df['count'], color='steelblue')
        plt.xlabel('Disease Class')
        plt.ylabel('Number of Predictions')
        plt.title('Prediction Distribution by Class')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/class_distribution.png', dpi=300)
        plt.close()
        
        # 3. Confusion matrix
        df = pd.read_sql_query('''
            SELECT p.predicted_class_name, 
                   COALESCE(f.correct_class_name, p.predicted_class_name) as true_class
            FROM predictions p
            JOIN feedback f ON p.id = f.prediction_id
        ''', conn)
        
        if not df.empty:
            from sklearn.metrics import confusion_matrix
            import numpy as np
            
            classes = sorted(df['predicted_class_name'].unique())
            cm = confusion_matrix(df['true_class'], df['predicted_class_name'], labels=classes)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300)
            plt.close()
        
        # 4. Confidence distribution
        df = pd.read_sql_query('''
            SELECT p.confidence, f.is_correct
            FROM predictions p
            JOIN feedback f ON p.id = f.prediction_id
        ''', conn)
        
        if not df.empty:
            plt.figure(figsize=(12, 6))
            plt.hist([df[df['is_correct'] == 1]['confidence'], 
                     df[df['is_correct'] == 0]['confidence']], 
                    bins=20, label=['Correct', 'Incorrect'], alpha=0.7)
            plt.xlabel('Confidence (%)')
            plt.ylabel('Frequency')
            plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{output_dir}/confidence_distribution.png', dpi=300)
            plt.close()
        
        conn.close()
        print(f"✅ Visualizations saved to {output_dir}/")
    
    def backup_database(self, backup_dir='backups'):
        """Create database backup"""
        import os
        import shutil
        
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(backup_dir, f'feedback_backup_{timestamp}.db')
        
        shutil.copy2(self.db_path, backup_file)
        print(f"✅ Database backed up to: {backup_file}")
        return backup_file
    
    def clean_old_data(self, days=90):
        """Remove predictions older than specified days (with no feedback)"""
        conn = self.get_connection()
        c = conn.cursor()
        
        c.execute('''DELETE FROM predictions 
                     WHERE id NOT IN (SELECT prediction_id FROM feedback)
                     AND timestamp < datetime('now', '-' || ? || ' days')''', (days,))
        
        deleted = c.rowcount
        conn.commit()
        conn.close()
        
        print(f"✅ Cleaned {deleted} old predictions without feedback")
        return deleted
    
    def print_report(self):
        """Print comprehensive database report"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("📊 DATABASE STATISTICS REPORT")
        print("=" * 60)
        
        print(f"\n📈 Overall Metrics:")
        print(f"   Total Predictions: {stats['total_predictions']}")
        print(f"   Total Feedback: {stats['total_feedback']}")
        print(f"   Correct: {stats['correct_predictions']}")
        print(f"   Incorrect: {stats['incorrect_predictions']}")
        print(f"   Accuracy: {stats['accuracy']:.2f}%")
        
        print(f"\n🎯 Training Queue:")
        print(f"   Pending: {stats['pending_training']}")
        print(f"   Used: {stats['used_in_training']}")
        
        print(f"\n📊 Prediction Distribution:")
        for class_name, count in stats['class_distribution'].items():
            print(f"   {class_name}: {count}")
        
        print(f"\n✅ Feedback by Class:")
        for class_name, counts in stats['feedback_by_class'].items():
            total = counts['correct'] + counts['incorrect']
            acc = (counts['correct'] / total * 100) if total > 0 else 0
            print(f"   {class_name}: {counts['correct']}/{total} ({acc:.1f}% accurate)")
        
        print("=" * 60 + "\n")


def main():
    """Interactive database management menu"""
    manager = DatabaseManager()
    
    while True:
        print("\n" + "=" * 60)
        print("🗄️  DATABASE MANAGEMENT MENU")
        print("=" * 60)
        print("1. View Statistics")
        print("2. Export to Excel")
        print("3. Generate Visualizations")
        print("4. Backup Database")
        print("5. Clean Old Data")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            manager.print_report()
        
        elif choice == '2':
            filename = input("Enter output filename (default: feedback_report.xlsx): ").strip()
            if not filename:
                filename = 'feedback_report.xlsx'
            manager.export_to_excel(filename)
        
        elif choice == '3':
            manager.generate_visualizations()
        
        elif choice == '4':
            manager.backup_database()
        
        elif choice == '5':
            days = input("Remove predictions older than how many days? (default: 90): ").strip()
            days = int(days) if days else 90
            manager.clean_old_data(days)
        
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
