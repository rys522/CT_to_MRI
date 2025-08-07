# MNIST Example: Comparison of Various Diffusion Methods
# Code for the Diffusion Study Group for LIST

# Recommende to install Ruff & Black extension in VScode

## Data preparation
run scripts/1_datadown.ipynb

## Training
python train.py --gpu {gpu}  # Other parameters can be set in GeneralConfig or ModelConfig
- ModelConfig is automatically saved in the checkpoint and loaded with it.

## Testing
python test.py --gpu {gpu}  # Other parameters can be set in TestConfig

## Author
Juhyung Park  
Laboratory for Imaging Science and Technology  
Seoul National University  
Email: jack0878@snu.ac.kr

# CT_to_MRI
