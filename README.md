# Predicting Age from Facial Images Using Deep Neural Networks

This repository contains the final project for the Machine Learning course, focused on building a robust pipeline for facial age prediction.


## Project Overview

Predicting a person’s age from a single facial image is a complex and impactful problem with real-world applications in healthcare, security, biometrics, and personalized user experiences. While humans rely on high-level cues, deep neural networks can detect fine-grained patterns—like skin texture, facial geometry, and wrinkles,that are not easily perceivable.

In this project, we developed an end-to-end deep learning pipeline using PyTorch to predict age from facial images. Our goal was to design a scalab and generalizable model that minimizes prediction error while maintaining stability across different data splits.

We began with a baseline CNN model, then incrementally enhanced performance using the following strategies:

-Data Cleaning and Preprocessing: Filtered invalid/missing entries, balanced gender distribution, and removed outliers

-Data Augmentation: Applied random crops, flips, rotations, and color transformations to boost generalization

-Transfer Learning: Leveraged pretrained ResNet-50 for deeper feature extraction and improved convergence

-Loss Function Optimization: Replaced MSE with Smooth L1 Loss for better resilience to outliers

-Cross-Validation & Hyperparameter Tuning: Ensured stable results using 5-fold CV and grid search optimization


We tracked performance using RMSE, R² score, and prediction vs. actual age plots to understand and refine model behavior.


Key Achievements:

Reduced RMSE from ~25 (baseline CNN) to ~7.5 with final model

Improved R² score from -2.02 to 0.79, indicating a strong correlation

Built a modular, reproducible, and scalable PyTorch-based pipeline


## Repository Structure

```
DeepLearning_FinalProject/
├── ML_Project_Report/                                          # Codebase and implementation for the age prediction pipeline
├── Model Performance Progression and Score Improvements/       # Model performance summaries
    
```

# Dataset

Source: Kaggle: Age Prediction (Spring'25 @ CU Denver)

Link: https://www.kaggle.com/competitions/age-prediction-spring-25-at-cu-denver/overview

Files:

wiki_labels.csv: Primary training data with image paths, age labels, gender, and face score

wiki_judge.csv: Test dataset metadata (no labels)

wiki_labeled/: Directory with ~60K facial images

wiki_judge_images/: Test images




