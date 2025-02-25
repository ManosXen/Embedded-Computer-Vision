# Embedded Computer Vision Project

This project was developed as part of the Embedded Systems course at the National Technical University of Athens (NTUA). It focuses on optimizing embedded computer vision through the fine-tuning of Convolutional Neural Networks (CNNs) for human/no-human recognition using the [Visual Wake Words Dataset](https://github.com/Mxbonn/visualwakewords). Additional techniques, such as post-training static and dynamic quantization, were applied to enhance model efficiency. In this project spliting images into multiple subframes was applied, in order to improving inference speed and enabling detection of multiple humans in a single image.

## Project Goals
- Fine-tune CNNs for accurate human/no-human classification.
- Implement quantization to optimize models for embedded systems.
- Enhance inference performance by processing image subframes.

## Project Structure
- **`fine_tune.ipynb`**: Jupyter Notebook demonstrating the CNN fine-tuning process using the Visual Wake Words Dataset.
- **`quantization.ipynb`**: Jupyter Notebook detailing the static and dynamic quantization steps (where applicable).
- **`run_scripts/`**: Directory containing utility scripts:
  - `check_models.py`: Script for splitting images into subframes and performing inference.
  - Additional scripts for streamlined model testing.
- **`observer_dataset/`**: A small subset of the Visual Wake Words Dataset used for deploying and testing quantized models.
- **`models/`**: Directory storing the state dictionaries of fine-tuned and quantized models.
