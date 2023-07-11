# Understanding MAML Through Its Loss Landscape
This repository contains the code base for the Master's Thesis of Bosong Ding. The project is dedicated to exploring and understanding the loss landscape of MAML (Model-Agnostic Meta-Learning). The training code is adopted from Chen et al.'s repository and visualization code is borrowed from Li et al.'s repository.

## Installation
To setup the necessary environment:

1. Clone the repository: `git clone <repo-link>`
2. Navigate to the cloned directory: `cd <cloned-repo-name>`
3. Install the requirements: `pip install -r requirements.txt`

## Getting Started
### CUB
To setup the CUB data:
1. Change your directory: `cd Training/filelists/CUB`
2. Run the download script: `source ./download_CUB.sh`

### mini-ImageNet
To setup the mini-ImageNet data:
1. Download the Kaggle dataset [here](https://www.kaggle.com/datasets/arjunashok33/miniimagenet)
2. Update `data_path` in `Training/filelists/miniImagenet/write_miniImagenet_filelist.py`
3. Run the setup script: `python Training/filelists/miniImagenet/write_miniImagenet_filelist.py`

## Training
Start the training process with the following command: `sh Training/run_train.sh `

## Testing
Run tests with the following command: `sh Training/run_test.sh`

## Results
The test results will be recorded in the `Training/record/` directory.

## Loss Landscape Visualization 
Visualize the loss landscape using: `sh Visualization/run_vis.sh`
To rescale the loss landscape, use: `sh Visualization/rescale.sh`

## Calculating Sharpness
Calculate the sharpness of the loss landscape using: `sh sharpness/run_sharpness.sh`

## References
1. [Chen et al.'s Repo](https://github.com/wyharveychen/CloserLookFewShot)
2. [Li et al.'s Repo](https://github.com/tomgoldstein/loss-landscape)

