import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import copy
from dataset import RadioSignalDataset
from model import RadioSignalCNN

def train_and_validate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting 5-Fold CV on device: {device}")
    csv_path = 'data/train.csv'
    img_path = 'data/train/'
    # full dataset mapping
    full_df = pd.read_csv(csv_path)
    # The targets are needed for Stratified K-Fold to balance the classes
    targets = full_df['label'].values 
    
    full_dataset = RadioSignalDataset(csv_file=csv_path, img_dir=img_path)
    # 5 folds/ 5 modele diferite pe chunk-uri de date training
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    # variabile folosite pt a vedea performanta fold-urilor
    fold_results = []
    # class weights for the slight imbalance
    # [class 1, class 2, class 3, class 4, class 5]
    weights_array = [0.8857, 1.0333, 1.0333, 1.0333, 1.0333]
    class_weights = torch.tensor(weights_array, dtype=torch.float32).to(device)
    
    # cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n--- Starting Fold {fold + 1}/{k_folds} ---")
        # ia subseturi de date pt fiecare fold 
        train_sub = Subset(full_dataset, train_idx)
        val_sub = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_sub, batch_size=32, shuffle=False, num_workers=2)
        
        # fiecare fold are un model diferit si este optimizat individual
        model = RadioSignalCNN(num_classes=5).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        best_val_acc = 0.0
        best_model_weights = None
        
        epochs = 15
        
        for epoch in range(epochs):
            model.train() # incepere parte de training 
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
            train_acc = (train_correct / train_total) * 100
            
            # incepere parte de validare
            model.eval() # partea de validare nu foloseste dropout si e scos la eval
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad(): # validare facuta fara gradient tracking pt a fi mai rapid
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
            val_acc = (val_correct / val_total) * 100
            
            print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            # salvare doar pt cea mai buna val a acuratetei
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                
        # cel mai bun model e salvat 
        torch.save(best_model_weights, f'cnn_fold_{fold+1}.pth')
        print(f"Fold {fold + 1} Best Validation Accuracy: {best_val_acc:.2f}% (Saved to cnn_fold_{fold+1}.pth)")
        fold_results.append(best_val_acc)
        
    print("\n=== 5-Fold Cross-Validation Complete ===")
    for i, acc in enumerate(fold_results):
        print(f"Fold {i+1}: {acc:.2f}%")
    print(f"Average CV Accuracy: {np.mean(fold_results):.2f}%")

if __name__ == "__main__":
    train_and_validate()