import os
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# read grayscale image and return HOG feature vector (HOG = histogram of oriented gradients, tine minte structura obiectelor in fct de schimbarile de luminozitate)
def extract_hog_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # read img
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # blur image so it has less noise
    img = cv2.medianBlur(img, 3)
    # HOG feature vector
    # orientations = 9 reprezinta 9 bucati pt unghiurile care ocupa 180 de grade (fiecare bucata reprezinta un slice de 20 de grade)
    # pixels_per_cell=(8, 8) inseamna ca imaginea e structurata in 8x8 patrate de pixeli 
    # cells_per_block=(2, 2) grupeaza in 2x2 grid (16x16 pixeli) pt normalizare
    # block_norm='L2-Hys' e formula matematica folosita pentru normalizare; limiteaza valorile maxime pentru a nu avea edge uri prea luminoase 
    features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features

# ia csv-ul si aplica extractia de tip hog pt fiecare imagine 
# returneaza X (features) si y (labels)
def prepare_data(csv_path, img_dir):
    df = pd.read_csv(csv_path)
    X = []
    y = []
    
    print(f"Extracting features for {len(df)} images. This may take a minute...")
    for index, row in df.iterrows():
        img_path = os.path.join(img_dir, str(row['id']))
        
        features = extract_hog_features(img_path)
        X.append(features)
        y.append(row['label'])
        
    return np.array(X), np.array(y)

if __name__ == "__main__":
    train_csv = 'data/train.csv'
    train_dir = 'data/train/'
    
    X, y = prepare_data(train_csv, train_dir)
    print(f"Feature matrix shape: {X.shape}") # 15500, number_of_hog_features
    
    # stratify=y ensures the 1-5 class balance is maintained in both splits
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training SVM Classifier")
    # C = regularization parameter
    # 1.0 default. 
    svm_model = SVC(kernel='rbf', C=1.0, random_state=42) 
    svm_model.fit(X_train, y_train)
    
    print("Predicting on validation set...")
    y_pred = svm_model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    print(f"Local Validation Accuracy: {acc * 100:.2f}%")
    
    cm = confusion_matrix(y_val, y_pred, labels=[1, 2, 3, 4, 5])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
    plt.title('SVM Baseline Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix_baseline.png') 

    # ---------------------------------------------------------
    # 7. Generate Submission File for the Leaderboard
    # ---------------------------------------------------------
    print("Generating predictions for the test set...")
    test_csv = 'data/test.csv'
    test_dir = 'data/test/'
    
    # Load test data
    test_df = pd.read_csv(test_csv)
    X_test = []
    
    # Extract features for test images
    for index, row in test_df.iterrows():
        img_path = os.path.join(test_dir, str(row['id']))
        features = extract_hog_features(img_path)
        X_test.append(features)
        
    X_test = np.array(X_test)
    
    # Predict using the trained SVM
    test_predictions = svm_model.predict(X_test)
    
    # Format the submission
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'label': test_predictions
    })
    
    # Save to CSV without the pandas index column (Mandatory for most formats)
    submission_df.to_csv('submission_baseline.csv', index=False)
    print("Submission file saved as 'submission_baseline.csv'. You can now upload this to the leaderboard!")