# text-recognizer -- Handwritten Paragraph Recognition System

## Overview
This project is a deep learning-based system designed to understand and interpret the content of handwritten paragraphs. Utilizing the modern stack of PyTorch and PyTorch-Lightning, the system integrates advanced deep learning architectures like CNNs, RNNs, and Transformers to effectively process and analyze handwritten text.

## Key Features
- **Modern Deep Learning Stack**: Built with PyTorch and PyTorch-Lightning for efficient model development and training.
- **Advanced Architectures**: Leverages CNNs, RNNs, and Transformers to handle the complexities of handwritten text recognition.
- **Experiment Management**: Utilizes Weights & Biases for comprehensive experiment tracking and management.
- **Continuous Integration**: Integrated with CircleCI to ensure robust, error-free code through continuous integration.
- **REST API for Predictions**: Features a FastAPI-based REST API, making the model accessible for inference through HTTP requests.
- **Deployment**: Deployed as a Docker container on AWS Lambda for scalable and efficient cloud-based service.
- **Data Monitoring**: Equipped with monitoring systems to alert for changes in the incoming data distribution, ensuring model reliability.

## Installation

```bash
# Clone the repository
git clone https://github.com/xuziyue/text-recognizer.git
cd text-recognizer

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python training/run_experiment.py --max_epochs=40 --gpus=1 --num_workers=8 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0.33 --model_class=LineCNNTransformer --window_width=20 --window_stride=12 --loss=transformer
```

### Experiments Tracking and Management
#### Setup wandb:
```bash
!wandb init
```
#### Track one experiment:
```bash
python training/run_experiment.py --wandb --max_epochs=40 --gpus=1 --num_workers=8 --data_class=EMNISTLines2 --model_class=LineCNNTransformer --loss=transformer
```
#### Hyperparameter tuning:
```bash
wandb sweep training/sweeps/emnist_lines2_line_cnn_transformer.yml
```
follow the returned instruction


## Continuous Integration
This project is integrated with CircleCI for automated testing and deployment. Check `.circleci/config.yml` for configuration details.

## Deployment
The system is containerized using Docker for deployment on AWS Lambda. Check the `Dockerfile` and deployment scripts in the `deploy` directory.
