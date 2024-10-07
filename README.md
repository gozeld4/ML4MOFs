# ML4MOFs

This repository contains code to find the optimal random state for data splitting across different machine learning models, specifically for predicting KVRH values of Metal-Organic Frameworks (MOFs). The machine learning models using different classifiers such as Random Forest, Gradient Boosting, Kernel Ridge Regression (KRR) with Laplacian kernel, and KRR with RBF kernel. The code uses four different datasets and calculates the R² score for each model.

### Includes:
- **ML_models_water_uptake_all_train.py**: Contains all the code for training the models and evaluating their performance.
- **run.sh**: A script to run the Python code on MIT's Supercloud.
- **create_csv_file.py**: The code for generating CSV files used as input.
- **combined_data_all.csv**: The dataset used for training and evaluation.

### Reference for Dataset:
1. Nandy, Aditya, et al. “A Database of Ultrastable MOFs Reassembled from Stable Fragments with Machine Learning Models.” *Matter*, vol. 6, no. 5, May 2023, pp. 1585–1603. [ScienceDirect](https://doi.org/10.1016/j.matt.2023.03.009).


