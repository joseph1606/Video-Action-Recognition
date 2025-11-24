- Video Classification with HMDB51
This project implements a video classification pipeline using the HMDB51 dataset. It leverages a
Long-term Recurrent Convolutional Network (LRCN) model that extracts spatial features from
individual video frames via a ResNet backbone and learns temporal dynamics through an
LSTM. The project includes scripts for preprocessing, training, and testing the model.

Table of Contents
- Dataset Preparation
- Environment Setup
- Preprocessing and Frame Extraction
- Training the Model
- Testing and Evaluation
- Project Structure
- Customization and Hyperparameters
  
Dataset Preparation
Step 0: Download and Unzip Dataset
1. Download Dataset:
Download the HMDB51 dataset from Kaggle. This dataset contains videos of 51 different
human action classes.
2. Unzip and Organize:
Unzip the downloaded dataset. The expected folder structure should be as follows:
- HMDB51
- Action_Class1
- Action_Class2
... ... ... ...
- Action_Class51
Each subdirectory represents a different action class.

Environment Setup
1. Python Version:
This project requires Python 3.7 or higher.
2. Dependencies:
Install the required Python packages by running:
# Requirements
pip install -r requirements.txt
## Key Libraries
- **PyTorch**
- **torchvision**
- **OpenCV**
- **scikit-learn**
- **tqdm**
- **numpy**
- **Pillow**
## Hardware Requirements
A CUDA-enabled GPU is recommended for training. The code automatically detects GPU
availability.
---

## Preprocessing and Frame Extraction
Before training, raw video files must be converted into frame sequences. The preprocessing
module includes functions for:
### Uniform Frame Sampling
- The `get_frames` function uses OpenCV to sample a fixed number of frames per video.
### Saving Frames to Disk
- The `store_frames` function writes the extracted frames as JPEG images.
Use these functions in your preprocessing notebook (e.g., `preprocessing.ipynb`) to convert all
videos into folders of extracted frames. The resulting folder structure should mirror the original
dataset structure.
---

## Training the Model
### Step 1: Run Training
Open the training notebook (e.g., `train_test_ucf50.ipynb`), and run the following command in a
cell to train the model:
!python run.py
--frame_dir "/path/to/UCF50_frames"
--train_size 0.75
--test_size 0.15
--model_type lrcn
--n_classes 50
--fr_per_vid 16
--batch_size 32
--n_epochs 20
--mode train
This command will:
- Load the frame dataset.
- Split data into training, validation, and test sets using stratified sampling.
- Apply data augmentation (resizing, random flips, affine transforms).
- Create PyTorch Datasets and DataLoaders.
- Initialize the LRCN model with a ResNet backbone.
- Set up the loss function, optimizer, and learning rate scheduler.
- Train the model while tracking loss and accuracy, saving the best model weights.
---

## Testing and Evaluation
### Step 2: Run Testing
In the testing notebook, run this command in a cell to evaluate the model:
!python run.py
--frame_dir "/path/to/UCF50_frames"
--mode eval
--ckpt "/path/to/models/best_model_wts.pt"
--n_classes 50
--batch_size 32
This will:
- Load dataset splits saved during training.
- Create a DataLoader for the test set.
- Load your trained model checkpoint.
- Evaluate the model on the test data.
- Compute overall accuracy, classification reports, and optionally confusion matrices.
---
## Customization and Hyperparameters
You can modify parameters in the commands above or notebook cells to experiment with
different settings:
### Data Parameters
- `--frame_dir`: Path to your preprocessed frames.
- `--fr_per_vid`: Number of frames sampled per video.
### Model Parameters
- `--model_type`: `'lrcn'` (default) or other supported types.
- `--cnn_backbone`: Options include `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`.
- `--rnn_hidden_size` and `--rnn_n_layers`: Configure LSTM network size.
### Training Parameters
- `--batch_size`, `--learning_rate`, `--n_epochs`, `--dropout`.
- `--train_size`, `--test_size`: Dataset split proportions.
---

## Summary of Steps
- **Step 0: Dataset Preparation**
Download, unzip, and organize the UCF50 dataset into subdirectories by action class.
- **Step 1: Run Training**
Use your notebook to invoke training with above command adjusting paths as needed.
- **Step 2: Run Testing**
Run evaluation similarly in notebook to assess model performance using your saved
checkpoint.
---
Happy Training!
