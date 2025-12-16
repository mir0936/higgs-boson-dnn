# CERN Higgs Boson Classification via Deep Neural Networks

## Overview
This project utilizes the official ATLAS Higgs Boson Machine Learning Challenge dataset to isolate the $H \rightarrow \tau \tau$ decay signal from overwhelming Standard Model background noise. 

## Architecture
1. **Data Pipeline:** Processes 30 kinematic features (transverse momenta, pseudo-rapidity, azimuthal angles, and reconstructed invariant masses) from 250,000 collision events.
2. **AI Classifier (PyTorch):** A Deep Neural Network (DNN) with dropout regularization designed to prevent overfitting to detector noise. The model outputs a continuous probability score for signal vs. background.

## Tech Stack
* **Deep Learning:** Python, PyTorch
* **Data Engineering:** Pandas, Scikit-Learn
* **Visualization:** Matplotlib

## Results
The model successfully converges using Binary Cross Entropy (BCE) loss. The resulting Receiver Operating Characteristic (ROC) curve demonstrates a high Area Under the Curve (AUC) score, proving the network's capability to effectively filter high-energy physics detector data.
