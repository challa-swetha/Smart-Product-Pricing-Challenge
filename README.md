
# Smart Product Pricing Challenge 🛍️

This project is a solution for the Smart Product Pricing Challenge. It uses a state-of-the-art multi-modal ensemble model to predict product prices based on their images and text descriptions.

## Methodology 🤖

This solution uses a hybrid model that leverages fine-tuned deep learning models for feature extraction, followed by a powerful gradient boosting ensemble. 

### Feature Engineering & Preprocessing

Three distinct feature sets were generated:

* **Vision Features (Fine-Tuned CNN):** An EfficientNetB3 model was fine-tuned on 75,000 product images to extract a 1536-dimension embedding vector for each image.
* **Textual Features (Fine-Tuned LLM):** A microsoft/deberta-v3-base Transformer model was fine-tuned for regression on 75,000 product descriptions to extract a 768-dimension embedding vector. 
* **Numerical Features:** Additional numerical features were also used in the model. 

### Final Predictive Model

A blended ensemble of two gradient boosting models was trained on the fused feature set:

1.  **LightGBM Regressor** 
2.  **XGBoost Regressor** (using `tree_method='gpu_hist'` for GPU acceleration)

The predictions from both models were blended using an optimized weighted average to achieve the lowest SMAPE score. 

## How to Run the Project 🚀

### 1. Set up the Environment

Clone this repository to your local machine:

git clone https://github.com/challa-swetha/Smart-Product-Pricing-Challenge.git
cd Smart-Product-Pricing-Challenge

### 2. Install Dependencies

Install all required packages:

pip install -r requirements.txt

### 3. Run the Pipeline

Open and run the solution notebook:

jupyter notebook solution.ipynb

Running all cells in the notebook will execute the full pipeline — feature extraction (vision + text embeddings), model training (LightGBM + XGBoost), and blending — and will generate `test_out.csv` containing the final predicted prices for the test set.
