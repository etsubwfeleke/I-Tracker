import numpy as np
import os
import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader

from DataLoader import iTrackerDataset
from iTrackerModel import iTracker

# try:
#     from iTrackerModel import iTracker
#     print("iTrackerData module imported successfully.")
# except ImportError:
#     print("iTrackerData module not found. Ensure it is in the same directory as this script.")
#     exit(1)
'''
Train/test code for iTracker.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/
Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''
DATA_PATH = '/Users/etsubfeleke/Documents/Pytorch_practice/iTracker/Gazecapture_processed'   

class Args:
    def __init__(self):
        self.data_path = DATA_PATH
        
args = Args()

num_workers = 2
num_epochs = 100
batch_size = 50
learning_rate = 0.001
momentum = 0.9 
weight_decay = 1e-4
best_prec_score = 1e20

count_test = 0
count = 0

# num_workers_values = [2, 4, 6, 8]
# for num_workers in num_workers_values:
#     print(f"Training with num_workers={num_workers}")

def calculate_gaze_error(output, gaze):
    cos_sim = torch.nn.functional.cosine_similarity(output, gaze, dim=1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    gaze_error = torch.acos(cos_sim) * (180.0 / np.pi) 
    return gaze_error.mean().item()

def main():
    global args, best_prec_score, weight_decay, momentum
    
    print("Initializing iTracker model...")
    model = iTracker()
    print("iTracker model initialized.")
    imsize = (224,224)
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using MPS backend for Apple Silicon GPU.')
    else:
        device = torch.device('cpu')
        print('Warning: No MPS found! Running on CPU.') 
        
    model.to(device)
    print(f"Model moved to {device} device.")
    
    training_dataset = iTrackerDataset(args.data_path, 'train', imsize)
    validation_dataset = iTrackerDataset(args.data_path, 'validation', imsize)
    test_dataset = iTrackerDataset(args.data_path, 'test', imsize)
    
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print("Test loader created")
    print("--------------------------------------------------------------------")
    
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        
        train_loss = 0.0
        for i, (row, left_eye, right_eye, face, faceGrid, gaze) in enumerate(training_loader):
            
            left_eye =left_eye.to(device)
            right_eye = right_eye.to(device)
            face = face.to(device)
            faceGrid = faceGrid.to(device)
            gaze = gaze.to(device)
            
            output = model(left_eye, right_eye, face, faceGrid)
            loss = criterion(output, gaze)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        model.eval()
        validation_loss = 0.0
        validation_gaze_error = 0
        total = 0
        
        with torch.no_grad():
            for i, (row, left_eye, right_eye, face, faceGrid, gaze) in enumerate(validation_loader):
                left_eye = left_eye.to(device)
                right_eye = right_eye.to(device)
                face = face.to(device)
                faceGrid = faceGrid.to(device)
                gaze = gaze.to(device)
                
                output = model(left_eye, right_eye, face, faceGrid)
                loss = criterion(output, gaze)
                validation_loss += loss.item()
                
                gaze_error = calculate_gaze_error(output, gaze)
                validation_gaze_error += gaze_error * gaze.size(0)
                total += gaze.size(0)
                
        validation_loss /= len(validation_loader)
        average_gaze_error = validation_gaze_error / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(training_loader):.4f},  Validation loss: {validation_loss:.4f},  Average Gaze Error: {average_gaze_error:.4f} degree')  
        
        if validation_loss < best_prec_score:
            best_prec_score = validation_loss
            torch.save({
                'epoch' : epoch + 1, 
                'state_dict' : model.state_dict(),
                'best_prec1' : best_prec_score, 
                'optimizer' : optimizer.state_dict() 
                }, 'checkpoint.pth.tar')
            print(f'Best Model Saved: (Loss: {best_prec_score:.4f})')
    print("Training Completed. Testing the model")
    
    try:
        checkpoint = torch.load('checkpoint.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded the best saved model from epoch{checkpoint['epoch']}")
    except:
        print(f"No checkpoint, using original model for testing")
    print("--------------------------------------------------------------------")       
    print("\nTesting the model...")
    test_model(model, test_loader, criterion, device=device)

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    testing_gaze_error = 0
    total = 0
    
    with torch.no_grad():
        for i, (row, left_eye, right_eye, face, faceGrid, gaze) in enumerate(test_loader):
            
            left_eye =left_eye.to(device)
            right_eye = right_eye.to(device)
            face = face.to(device)
            faceGrid = faceGrid.to(device)
            gaze = gaze.to(device)
            
            output = model(left_eye, right_eye, face, faceGrid)
            loss = criterion(output, gaze)
            test_loss += loss.item()
            
            gaze_error = calculate_gaze_error(output, gaze)
            testing_gaze_error += gaze_error * gaze.size(0)
            total += gaze.size(0)
            
        
    test_loss /= len(test_loader)
    average_testing_gaze_error = testing_gaze_error / total
    
    print(f'Testing Results:')
    print(f'Test Loss: {test_loss:.4f}, Average Gaze Error: {average_testing_gaze_error:.4f} degree')
    print(f'Testing Done')
        
if __name__ == '__main__':
    main()