import torch
import torch.nn as nn
import torch.nn.functional as F

class RadioSignalCNN(nn.Module): # mosteneste din clasa oficiala a pytorch-ului pt a lua mutarea pe GPU, salvarea weight-urilor intr-un file si observarea gradientelor
    # blocurile sunt pastrate separat una de alta in init
    def __init__(self, num_classes=5): # 3 stagii de procesare/ blocuri (daca luam prea multe ajungeam in overfitiing, modelul doar memora complet datele de training)
        super(RadioSignalCNN, self).__init__()
        # img 128x55
        # prima etapa de procesare ia pixelii si gaseste drepte, curbe sau edge-uri (3x3 pixels wide)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # la fiecare apel a fct MaxPool2d imaginea se micsoreaza 
        # img devine 64x27
        # a doua etapa combina liniile si curbele simple din prima si formeaza forme geometrice simple (tip cercuri)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # (6x6 pixels wide fiindca imaginea a fost "taiata")
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # img 32x13 
        # ultima etapa este cea care chiar formeaza semnalele radio din formele geometrice gasite de etapa 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # ultimul cut duce imaginea la 16x6 pixeli
        # in loc de flattening  
        # se foloseste global average pooling asa ca modelul cauta average value-ul pt fiecare channel de pixeli = raman exact 128 de numere fiecare reprezentand cat de puternic apare feature-ul ala in imagine
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # previne overfitting-ul in timpul training-ului (dropout-ul se intampla doar la training) 
        self.dropout = nn.Dropout(p=0.4) # sunt scoase random 40% din feature-uri la fiecare batch pt ca modelul sa nu se axeze mereu pe feature-urile obvious  
        # layer final de output cu 128 channels si 5 classes
        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # trecerea imaginii prin blocurile de procesare
        x = self.pool1(F.relu(self.bn1(self.conv1(x)))) # block 1
        x = self.pool2(F.relu(self.bn2(self.conv2(x)))) # block2
        x = self.pool3(F.relu(self.bn3(self.conv3(x)))) # block 3
        
        # unificare si flatten
        x = self.global_pool(x)
        x = torch.flatten(x, 1) # Flattens from [Batch, 128, 1, 1] to [Batch, 128]

        # clasificare
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
# testare pt sanity check pe un singur batch
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset import RadioSignalDataset # dataset-ul custom
    import torch.optim as optim

    print("Initializing Single Batch Overfitting Test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # setup ca sa ia GPU (pe laptop nu am dar am rulat asta intr-un )
    print(f"Using device: {device}")

    train_dataset = RadioSignalDataset(csv_file='data/train.csv', img_dir='data/train/')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # 32 de imagini intr-un singur batch
    
    # Grab the first batch and move it to the device
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    # 3. Initialize Model, Loss, and Optimizer
    model = RadioSignalCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    # AdamW includes weight decay (L2 regularization) natively
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 4. The Training Loop (50 Epochs on the SAME batch)
    model.train() # Put model in training mode
    
    for epoch in range(50):
        # A. Zero the gradients
        optimizer.zero_grad()
        
        # B. Forward pass: make predictions
        outputs = model(images)
        
        # C. Calculate the loss
        loss = criterion(outputs, labels)
        
        # D. Backward pass: calculate gradients
        loss.backward()
        
        # E. Update weights
        optimizer.step()
        
        # Calculate accuracy for this batch
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = (correct / labels.size(0)) * 100

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/50] | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")

    print("Sanity check complete. If Accuracy reached 100% and Loss is near 0.0, your architecture is verified!")