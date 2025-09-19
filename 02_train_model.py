import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

def train_model():
    """
    Reads the collected gesture data, trains a RandomForestClassifier,
    evaluates its performance, and saves the trained model.
    """
    # Load the dataset
    try:
        df = pd.read_csv('gestures.csv')
    except FileNotFoundError:
        print("Error: 'gestures.csv' not found. Please run the data collection script first.")
        return

    if df.empty:
        print("Error: 'gestures.csv' is empty. Please collect some data.")
        return

    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")
    print("Gesture counts:\n", df['label'].value_counts())


    # Prepare the data
    X = df.drop('label', axis=1)  # Features (landmark coordinates)
    y = df['label']              # Target (gesture labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining model...")
    # Initialize and train the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

    # Display confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print(cm)
    
    # Optional: Visualize the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


    # Save the trained model to a file
    model_filename = 'gesture_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"\nModel saved successfully as '{model_filename}'")

if __name__ == '__main__':
    train_model()
