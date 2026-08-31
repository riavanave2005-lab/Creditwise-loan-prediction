import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score

class ModelEvaluator:
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        if y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
        self.results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        return metrics
    
    def evaluate_all_models(self, models, X_test, y_test):
        print("\n=== Model Evaluation ===")
        for name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, name)
            print(f"\n📊 {name}:")
            print(f"   Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall:    {metrics['recall']:.4f}")
            print(f"   F1-Score:  {metrics['f1']:.4f}")
            if 'auc_roc' in metrics:
                print(f"   AUC-ROC:   {metrics['auc_roc']:.4f}")
        best_name = max(self.results, key=lambda x: self.results[x]['metrics']['accuracy'])
        print("\n" + "="*50)
        print(f"🏆 Best Performing Model: {best_name}")
        print(f"   Test Accuracy: {self.results[best_name]['metrics']['accuracy']:.4f}")
        return best_name
    
    def plot_confusion_matrices(self, save_path='models/confusion_matrices.png'):
        n_models = len(self.results)
        if n_models == 0:
            return
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        for idx, (name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name}\nConfusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Confusion matrices saved to {save_path}")
    
    def plot_model_comparison(self, save_path='models/model_comparison.png'):
        metrics_df = pd.DataFrame()
        for name, result in self.results.items():
            metrics = result['metrics']
            metrics_df[name] = pd.Series(metrics)
        metrics_df.plot(kind='bar', figsize=(10, 6))
        plt.title('Model Performance Comparison')
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Model comparison saved to {save_path}")
