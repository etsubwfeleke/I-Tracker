# iTracker Training Results

## Dataset Statistics
- Training Set: 9,089 records
- Validation Set: 9,089 records
- Test Set: 2,682 records

## Training Progress

### Key Milestones
1. **Initial Performance (Epoch 1)**
   - Training Loss: 33.37
   - Validation Loss: 31.15
   - Gaze Error: 55.89°

2. **Mid-Training (Epoch 50)**
   - Training Loss: ~4.5
   - Validation Loss: ~2.5
   - Gaze Error: ~10°

3. **Best Model (Epoch 95)**
   - Training Loss: 3.04
   - Validation Loss: 1.66
   - Gaze Error: 8.18°

4. **Final Model (Epoch 100)**
   - Training Loss: 3.09
   - Validation Loss: 2.12
   - Gaze Error: 8.79°

## Test Set Evaluation
- Test Loss: 51.82
- Average Gaze Error: 53.63°

## Hardware Configuration
- Platform: Apple Silicon
- Backend: MPS (Metal Performance Shaders)
- Batch Size: 50
- Learning Rate: 0.001
- Weight Decay: 1e-4

## Notable Observations
- Rapid initial convergence (epochs 1-10)
- Stable validation performance in later epochs
- Significant gap between validation and test performance
- Best model saved at epoch 95