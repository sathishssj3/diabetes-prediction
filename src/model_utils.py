import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_model(X_train, y_train, model_type='logistic_regression'):
    """
    Trains a classification model based on the specified model type.
    """
    if model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42)
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
    else:
        raise ValueError("Unsupported model type. Choose from: 'logistic_regression', 'decision_tree', 'random_forest'")
        
    print(f"Training {model_type.replace('_', ' ').title()} model...")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints accuracy, confusion matrix, and classification report.
    """
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)

def save_model(model, model_type):
    """
    Saves the trained model to disk.
    """
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_path = os.path.join('models', f'{model_type}_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def load_saved_model(model_path):
    """
    Loads a saved model from disk.
    """
    return joblib.load(model_path)
