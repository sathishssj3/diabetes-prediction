from src.data_utils import load_data, preprocess_data
from src.model_utils import train_model, evaluate_model, save_model

def main():
    print("1. Loading Data...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    
    print("\n2. Preprocessing Data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print(f"Training data size: {X_train.shape[0]} samples")
    print(f"Test data size: {X_test.shape[0]} samples")
    
    print("\n3. Training, Evaluating, and Saving Models...")
    
    models_to_train = ['logistic_regression', 'decision_tree', 'random_forest']
    
    for model_type in models_to_train:
        print(f"\n==================================================")
        print(f"  Evaluating {model_type.replace('_', ' ').title()}")
        print(f"==================================================")
        
        # Train
        model = train_model(X_train, y_train, model_type=model_type)
        
        # Evaluate
        print("\n--- Model Evaluation ---")
        evaluate_model(model, X_test, y_test)
        
        # Save
        save_model(model, model_type)

if __name__ == "__main__":
    main()
