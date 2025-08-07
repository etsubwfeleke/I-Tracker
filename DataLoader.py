import torch.utils.data as data
import scipy.io as sio
from PIL import Image
import os 
import os.path
import torchvision.transforms as transforms
import torch 
import numpy as np
import re
from torch.utils.data import Dataset


'''
Data loader for the iTracker.
Use prepareDataset.py to convert the dataset from http://gazecapture.csail.mit.edu/ to proper format.

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


def loadMetadata(filename, silent = False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata

class scalarmean (object):
    """Normalize tensor"""
    
    def __init__(self, mean , std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1,1,1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1,1,1)
        
    def __call__(self, tensor):
        return (tensor - self.mean) / self.std
    
    
class iTrackerDataset(Dataset):
    def __init__(self,data_dir, split= 'train', imsize=(224,224), gridsize=(25,25)):
        self.data_dir = data_dir
        self.imsize = imsize
        self.gridsize =gridsize
        
        # From ITrackerData.py
        # """"""""""""""
        print('Loading iTracker dataset...')
        metaFile = os.path.join(data_dir, 'metadata.mat')
        #metaFile = 'metadata.mat'
        if metaFile is None or not os.path.isfile(metaFile):
            raise RuntimeError('There is no such file %s! Provide a valid dataset path.' % metaFile)
        self.metadata = loadMetadata(metaFile)
        if self.metadata is None:
            raise RuntimeError('Could not read metadata file %s! Provide a valid dataset path.' % metaFile)
        # """"""""""""""""
        
        self.transformLefteye = transforms.Compose([
            transforms.Resize(self.imsize),
            transforms.ToTensor(),
            scalarmean(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.transformRighteye = transforms.Compose([
            transforms.Resize(self.imsize),
            transforms.ToTensor(),
            scalarmean(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.transformFace = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            scalarmean(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        # From ITrackerData.py
        # """""""""""""
        if split == 'test':
            mask = self.metadata['labelTest']
        elif split == 'val':
            mask = self.metadata['labelVal']
        else:
            mask = self.metadata['labelTrain']

        self.indices = np.argwhere(mask)[:,0]
        print('Loaded iTracker dataset split "%s" with %d records...' % (split, len(self.indices)))
        # """""""""""""
        
    def imageloader(self, path):
            image = Image.open(path).convert('RGB')
            return image
        
    def generate_face_grid(self, face_grid_metadata, grid_size=25):
    
            grid = np.zeros((grid_size, grid_size), dtype=int)
            
            grid_x = int(face_grid_metadata[0] * grid_size)
            grid_y = int(face_grid_metadata[1] * grid_size)
            grid_w = int(face_grid_metadata[2] * grid_size)
            grid_h = int(face_grid_metadata[3] * grid_size)
            
            x1 = max(0, grid_x)
            y1 = max(0, grid_y)
            x2 = min(grid_size - 1, grid_x + grid_w - 1)
            y2 = min(grid_size - 1, grid_y + grid_h - 1)
            
            grid[y1:y2+1, x1:x2+1] = 1
            
            return torch.tensor(grid, dtype=torch.float32)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
            index = self.indices[idx]
            
            left_eye_paths = os.path.join(self.data_dir, '%05d/appleLeftEye/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
            right_eye_paths = os.path.join(self.data_dir, '%05d/appleRightEye/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
            face_paths = os.path.join(self.data_dir, '%05d/appleFace/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
            
            
            left_eye = self.imageloader(left_eye_paths)
            right_eye = self.imageloader(right_eye_paths)
            face = self.imageloader(face_paths)
            
            left_eye = self.transformLefteye(left_eye)
            right_eye = self.transformRighteye(right_eye)
            face = self.transformFace(face)
            
            # From ITrackerData.py
            # """""""""""""
            gaze = np.array([self.metadata['labelDotXCam'][index], self.metadata['labelDotYCam'][index]], np.float32)
            # """""""""""""
            
            
            
            faceGrid = self.generate_face_grid(self.metadata['labelFaceGrid'][index,:])
            
            
            # From ITrackerData.py
            # """""""""""""
            row = torch.LongTensor([int(index)])
            faceGrid = torch.FloatTensor(faceGrid)
            gaze = torch.FloatTensor(gaze)
            # """""""""""""
        
        
            return row, left_eye, right_eye, face, faceGrid, gaze
    

# -----------------------------------------
# Test the dataloader   
# -----------------------------------------
if __name__ == "__main__":
    dataset = iTrackerDataset(data_dir='iTracker/Gazecapture_processed', split='train')
    print(f"Dataset size: {len(dataset)}")
    
    for i in range(5):
        row, left_eye, right_eye, face, face_grid, gaze = dataset[i]
        print(f"Sample {i}:")
        print(f"Row: {row}")
        print(f"Left Eye Shape: {left_eye.shape}")
        print(f"Right Eye Shape: {right_eye.shape}")
        print(f"Face Shape: {face.shape}")
        print(f"Face Grid Shape: {face_grid.shape}")
        print(f"Gaze Target: {gaze}")
