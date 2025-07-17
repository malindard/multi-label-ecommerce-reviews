import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, multilabel_confusion_matrix, accuracy_score
from sklearn.metrics import hamming_loss, jaccard_score
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AspectClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.label_columns = None
        
    def load_data(self, train_file='data/processed/train_data.csv', 
                  val_file='data/processed/val_data.csv',
                  test_file='data/processed/test_data.csv'):
        """Load preprocessed data"""
        logger.info("Loading preprocessed data...")
        
        self.train_df = pd.read_csv(train_file)
        self.val_df = pd.read_csv(val_file)
        self.test_df = pd.read_csv(test_file)
        
        # Get label columns
        self.label_columns = [col for col in self.train_df.columns if col.startswith('has_')]
        
        logger.info(f"Training data: {len(self.train_df)} samples")
        logger.info(f"Validation data: {len(self.val_df)} samples")
        logger.info(f"Test data: {len(self.test_df)} samples")
        logger.info(f"Label columns: {self.label_columns}")
        
        return self.train_df, self.val_df, self.test_df
    
    def prepare_features(self, max_features=5000, ngram_range=(1, 2)):
        """Prepare text features using TF-IDF"""
        logger.info("Preparing text features...")
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=None,  # Keep all words for Indonesian
            lowercase=True,
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        
        # Combine all text data for fitting
        all_text = pd.concat([
            self.train_df['clean_text'],
            self.val_df['clean_text'],
            self.test_df['clean_text']
        ])
        
        # Fit vectorizer on all data
        self.vectorizer.fit(all_text)
        
        # Transform datasets
        self.X_train = self.vectorizer.transform(self.train_df['clean_text'])
        self.X_val = self.vectorizer.transform(self.val_df['clean_text'])
        self.X_test = self.vectorizer.transform(self.test_df['clean_text'])
        
        # Prepare labels
        self.y_train = self.train_df[self.label_columns]
        self.y_val = self.val_df[self.label_columns]
        self.y_test = self.test_df[self.label_columns]
        
        logger.info(f"Feature matrix shape: {self.X_train.shape}")
        logger.info(f"Label matrix shape: {self.y_train.shape}")
        
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
    
    def train_model(self, model_type='logistic_regression'):
        """Train multi-label classifier"""
        logger.info(f"Training {model_type} model...")
        
        # Choose base classifier
        if model_type == 'logistic_regression':
            base_classifier = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0,
                class_weight='balanced'
            )
        elif model_type == 'random_forest':
            base_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                class_weight='balanced'
            )
        elif model_type == 'svm':
            base_classifier = SVC(
                random_state=42,
                probability=True,
                kernel='rbf',
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Multi-output classifier
        self.model = MultiOutputClassifier(base_classifier, n_jobs=-1)
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        logger.info("Model training completed!")
        return self.model
    
    def evaluate_model(self, X, y, dataset_name="Dataset"):
        """Evaluate model performance"""
        logger.info(f"Evaluating model on {dataset_name}...")
        
        # Make predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        hamming = hamming_loss(y, y_pred)
        jaccard = jaccard_score(y, y_pred, average='weighted')
        
        print(f"\n=== {dataset_name} Performance ===")
        print(f"Exact Match Accuracy: {accuracy:.4f}")
        print(f"Hamming Loss: {hamming:.4f}")
        print(f"Jaccard Score: {jaccard:.4f}")
        
        # Per-label classification report
        print(f"\n=== Per-Label Classification Report ({dataset_name}) ===")
        for i, label in enumerate(self.label_columns):
            print(f"\n{label}:")
            print(classification_report(y.iloc[:, i], y_pred[:, i], 
                                      target_names=['Not Present', 'Present']))
        
        return {
            'accuracy': accuracy,
            'hamming_loss': hamming,
            'jaccard_score': jaccard,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def plot_label_distribution(self):
        """Plot label distribution across datasets"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, label in enumerate(self.label_columns):
            ax = axes[i]
            
            # Count occurrences
            train_count = self.y_train.iloc[:, i].sum()
            val_count = self.y_val.iloc[:, i].sum()
            test_count = self.y_test.iloc[:, i].sum()
            
            # Plot
            datasets = ['Train', 'Validation', 'Test']
            counts = [train_count, val_count, test_count]
            
            ax.bar(datasets, counts, color=['blue', 'orange', 'green'])
            ax.set_title(f'{label.replace("has_", "").title()}')
            ax.set_ylabel('Count')
            
            # Add percentage labels
            total_train = len(self.y_train)
            total_val = len(self.y_val)
            total_test = len(self.y_test)
            
            for j, (dataset, count, total) in enumerate(zip(datasets, counts, [total_train, total_val, total_test])):
                percentage = (count / total) * 100
                ax.text(j, count + 0.01 * max(counts), f'{percentage:.1f}%', 
                       ha='center', va='bottom')
        
        # Remove empty subplot
        if len(self.label_columns) < 6:
            fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig('models/label_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, top_n=10):
        """Plot feature importance for each aspect"""
        if not hasattr(self.model, 'estimators_'):
            logger.warning("Feature importance not available for this model type")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        for i, (label, estimator) in enumerate(zip(self.label_columns, self.model.estimators_)):
            ax = axes[i]
            
            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
            elif hasattr(estimator, 'coef_'):
                importances = np.abs(estimator.coef_[0])
            else:
                continue
            
            # Get top features
            top_indices = np.argsort(importances)[-top_n:]
            top_features = [feature_names[idx] for idx in top_indices]
            top_importances = importances[top_indices]
            
            # Plot
            ax.barh(range(len(top_features)), top_importances)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features)
            ax.set_title(f'{label.replace("has_", "").title()} - Top Features')
            ax.set_xlabel('Importance')
        
        # Remove empty subplot
        if len(self.label_columns) < 6:
            fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model_path='models/aspect_classifier.pkl', 
                   vectorizer_path='models/tfidf_vectorizer.pkl'):
        """Save trained model and vectorizer"""
        logger.info("Saving model and vectorizer...")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save label columns
        with open('models/label_columns.txt', 'w') as f:
            for label in self.label_columns:
                f.write(f"{label}\n")
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Vectorizer saved to {vectorizer_path}")
        
        return model_path, vectorizer_path
    
    def cross_validate(self, cv_folds=5):
        """Perform cross-validation"""
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        # Combine train and validation for CV
        X_combined = np.vstack([self.X_train.toarray(), self.X_val.toarray()])
        y_combined = pd.concat([self.y_train, self.y_val])
        
        # Cross-validation for each label
        cv_results = {}
        
        for i, label in enumerate(self.label_columns):
            base_classifier = LogisticRegression(random_state=42, max_iter=1000)
            scores = cross_val_score(base_classifier, X_combined, 
                                   y_combined.iloc[:, i], cv=cv_folds, 
                                   scoring='f1_weighted')
            cv_results[label] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
        
        # Print results
        print(f"\n=== Cross-Validation Results ({cv_folds}-fold) ===")
        for label, result in cv_results.items():
            print(f"{label}: {result['mean_score']:.4f} (+/- {result['std_score']*2:.4f})")
        
        return cv_results

def main():
    """Main training pipeline"""
    # Initialize classifier
    classifier = AspectClassifier()
    
    # Load data
    train_df, val_df, test_df = classifier.load_data()
    
    # Prepare features
    X_train, y_train, X_val, y_val, X_test, y_test = classifier.prepare_features()
    
    # Plot label distribution
    classifier.plot_label_distribution()
    
    # Train model
    model = classifier.train_model(model_type='logistic_regression')
    
    # Evaluate on different datasets
    train_results = classifier.evaluate_model(X_train, y_train, "Training")
    val_results = classifier.evaluate_model(X_val, y_val, "Validation")
    test_results = classifier.evaluate_model(X_test, y_test, "Test")
    
    # Cross-validation
    cv_results = classifier.cross_validate()
    
    # Plot feature importance
    classifier.plot_feature_importance()
    
    # Save model
    model_path, vectorizer_path = classifier.save_model()
    
    # Training summary
    print(f"\n=== Training Summary ===")
    print(f"Model: Logistic Regression with Multi-Output")
    print(f"Features: TF-IDF (max_features=5000, ngram_range=(1,2))")
    print(f"Labels: {len(classifier.label_columns)} aspects")
    print(f"Training accuracy: {train_results['accuracy']:.4f}")
    print(f"Validation accuracy: {val_results['accuracy']:.4f}")
    print(f"Test accuracy: {test_results['accuracy']:.4f}")
    
    return classifier, test_results

if __name__ == "__main__":
    classifier, results = main()