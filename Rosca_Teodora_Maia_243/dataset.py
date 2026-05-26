import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class RadioSignalDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        explicatie atribute:
            csv_file (string): path to the csv file with annotations
            img_dir (string): directory with data
            transform (callable, optional): transformation that might be applied to sample
        """
        self.data_frame = pd.read_csv(csv_file) # pandas loads csv files into dataframes 
        self.data_frame['filename'] = self.data_frame['id'].astype(str)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx]['filename'])
        
        # read image using OpenCV 
        # radio signals arent changed by img colour so grayscale is sufficient
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        
        # opencv reads images as numpy arrays
        if self.transform:
            image = self.transform(image)
        else:
            # default transformation = convert to tensor and add channel dimension
            image = torch.from_numpy(image).float().unsqueeze(0) / 255.0

        label = int(self.data_frame.iloc[idx]['label'])
        label = label - 1  # bc cross entropy loss works from 0-4 labels go down 1 (they were 1-5) 

        return image, label

if __name__ == "__main__":
    csv_path = 'data/train.csv'
    img_path = 'data/train/'
    
    train_dataset = RadioSignalDataset(csv_file=csv_path, img_dir=img_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2) # data loader = fetches items shuffled bc neural networks learn best with small random data not with one by one chunks
    
    # fetch just the first batch to verify shapes
    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")
    print(f"Sample Labels (0-indexed): {labels[:5]}")