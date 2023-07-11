This is the code for Bosong Ding's Master Thesis, "Understanding MAML Through Its Loss Landscape".
The training code is drawn from Chen et al.'s repo: https://github.com/wyharveychen/CloserLookFewShot
The visualization code is drawn from Li et al.'s repo: https://github.com/tomgoldstein/loss-landscape

## Environment
`pip install requirements.txt`

## Getting started
### CUB
* Change directory to `Training/filelists/CUB`
* run `source ./download_CUB.sh`

### mini-ImageNet
download Kaggle datasethttps://www.kaggle.com/datasets/arjunashok33/miniimagenet
change `data_path` in `Training/filelists/miniImagenet/write_miniImagenet_filelist.py`
`python Training/filelists/miniImagenet/write_miniImagenet_filelist.py`

## Train
Run
`sh Training/run_train.sh `

## Test
`sh Training/run_test.sh`

## Results
The test results will be recorded in `./record/`

## loss landscape visualization 
`sh Visualization/run_vis.sh`
to rescale the loss landscape: `sh Visualization/rescale.sh`

## Calculate sharpness
`sh sharpness/run_sharpness.sh`


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
The test results will be recorded in the `./record/` directory.

## Loss Landscape Visualization 
Visualize the loss landscape using: `sh Visualization/run_vis.sh`
To rescale the loss landscape, use: `sh Visualization/rescale.sh`

## Calculating Sharpness
Calculate the sharpness of the loss landscape using: `sh sharpness/run_sharpness.sh`

## References
1. [Chen et al.'s Repo](https://github.com/wyharveychen/CloserLookFewShot)
2. [Li et al.'s Repo](https://github.com/tomgoldstein/loss-landscape)

## License
This project is licensed under the terms of the MIT license.
