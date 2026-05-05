"""
Complete End-to-End Image Classification Pipeline
This script covers data loading, model training, validation, and final inference.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==========================================
# BLOCK 1: Configuration & Setup
# ==========================================
class Config:
    IMG_DIR = './data/images'
    TRAIN_CSV = './data/train.csv'
    TEST_CSV = './data/test.csv'
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 0.0001
    NUM_CLASSES = 20
    # Automatically choose GPU if available, otherwise use CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# BLOCK 2: Data Processing
# ==========================================
class CustomImageDataset(Dataset):
    """Loads images and labels from a CSV file."""
    def __init__(self, dataframe, img_dir, transform=None, is_test=False):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Construct image path and load image
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB') 
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_test:
            # If testing, we don't have labels, just return image and its ID
            return image, self.data.iloc[idx, 0]
        
        # If training, extract the label (assuming one-hot encoded in CSV)
        label_row = self.data.iloc[idx, 1:].values.astype('float32')
        label_idx = np.argmax(label_row) 
        return image, torch.tensor(label_idx, dtype=torch.long)

def get_dataloaders():
    """Sets up image augmentations and returns training, validation, and test dataloaders."""
    # Define how we want to alter the images before feeding them to the model
    transform_pipeline = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataframes
    full_train_df = pd.read_csv(Config.TRAIN_CSV)
    train_df, val_df = train_test_split(full_train_df, test_size=0.2, random_state=42)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Create Datasets
    train_dataset = CustomImageDataset(train_df, Config.IMG_DIR, transform=transform_pipeline)
    val_dataset = CustomImageDataset(val_df, Config.IMG_DIR, transform=transform_pipeline)
    test_dataset = CustomImageDataset(test_df, Config.IMG_DIR, transform=transform_pipeline, is_test=True)

    # Create DataLoaders to batch the data
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

# ==========================================
# BLOCK 3: Model Initialization
# ==========================================
def build_model():
    """Loads a pre-trained model and modifies the final layer for our classes."""
    model = models.densenet121(weights='IMAGENET1K_V1')
    # Change the final classification layer to match our number of classes
    model.classifier = nn.Linear(model.classifier.in_features, Config.NUM_CLASSES)
    return model.to(Config.DEVICE)

# ==========================================
# BLOCK 4: Training Engine
# ==========================================
def train_model(model, train_loader, val_loader):
    """Trains the model and evaluates it on the validation set."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    best_loss = float('inf')

    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Training Loop
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]"):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()           # Clear old gradients
            outputs = model(images)         # Forward pass
            loss = criterion(outputs, labels) # Calculate error
            loss.backward()                 # Backpropagation
            optimizer.step()                # Update weights
            
            running_loss += loss.item()
            
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]"):
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save the best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("--> Saved new best model!")

# ==========================================
# BLOCK 5: Inference & Submission
# ==========================================
def generate_predictions(model, test_loader):
    """Loads the best model, runs inference on test data, and saves a CSV."""
    print("Loading best model for inference...")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    submission_data = []

    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc="Generating Predictions"):
            images = images.to(Config.DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Format predictions (Example: converting back to one-hot encoding)
            for i in range(len(image_ids)):
                img_id = image_ids[i]
                pred_idx = predicted[i].item()
                
                one_hot = [0] * Config.NUM_CLASSES
                one_hot[pred_idx] = 1 
                row = [img_id] + one_hot
                submission_data.append(row)

    # Read the original columns to format the CSV correctly
    columns = pd.read_csv(Config.TRAIN_CSV, nrows=0).columns
    submission_df = pd.DataFrame(submission_data, columns=columns)
    submission_df.to_csv('final_submission.csv', index=False)
    print("Predictions saved to final_submission.csv!")

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    print(f"Starting pipeline on device: {Config.DEVICE}")
    train_loader, val_loader, test_loader = get_dataloaders()
    
    model = build_model()
    
    # Step 1: Train the model
    train_model(model, train_loader, val_loader)
    
    # Step 2: Generate final predictions on unseen data
    generate_predictions(model, test_loader)