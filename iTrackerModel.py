import torch.nn as nn
import torch


class iTracker(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(iTracker, self).__init__()
        
        self.left_eye_branch = nn.Sequential(
            nn.Conv2d(3, 96 , kernel_size=11 , stride=4, padding=0), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=1, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(384 , 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
        )
        
        self.left_eye_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(), 
            nn.Linear(128, 64)
        )
        
        self.right_eye_branch = nn.Sequential(
            nn.Conv2d(3, 96 , kernel_size=11 , stride=4, padding=0), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=1, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(384 , 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.right_eye_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(), 
            nn.Linear(128, 64)
        )
        
        self.face_branch = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.face_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128,128),
            nn.ReLU()
        )
        self.face_grid_branch = nn.Sequential(
            nn.Linear(25 * 25, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fully_con_layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(64 + 64 + 128 + 128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)
        )
    def forward(self, left_eye, right_eye, face, face_grid):
                # ...
        #print(f"DEBUG: Before calling left_eye_branch, left_eye device: {left_eye.device}") # Add this
        left_eye_load = self.left_eye_branch(left_eye)
        #left_eye_load = left_eye_load.view(left_eye_load.size(0), -1)
        left_eye_load = self.left_eye_fc(left_eye_load)
        
        right_eye_load = self.right_eye_branch(right_eye)
        #right_eye_load = right_eye_load.view(right_eye_load.size(0), -1)
        right_eye_load = self.right_eye_fc(right_eye_load)
        
        face_load = self.face_branch(face)
        #face_load = face_load.view(face_load.size(0), -1)
        face_load = self.face_fc(face_load)
        
        flat_face_grid = face_grid.view(face_grid.size(0), -1)
        face_grid_features = self.face_grid_branch(flat_face_grid)
        
                
        concatenate_features = torch.cat([
            left_eye_load,
            right_eye_load,
            face_load,
            face_grid_features
        ], dim=1)
        
        gaze_target = self.fully_con_layers(concatenate_features)
        
        return gaze_target
    
# -----------------------------------------
# Test the model
# -----------------------------------------
if __name__ == "__main__":
    model = iTracker()
    left_eye = torch.rand(10, 3, 224, 224)  
    right_eye = torch.rand(10, 3, 224, 224)  
    face = torch.rand(10, 3, 224, 224)      
    face_grid = torch.rand(10, 25, 25)      
    
    output = model(left_eye, right_eye, face, face_grid)
    print(f"Output shape: {output.shape}")  