
# Smart Product Pricing Challenge 🛍️

This project is a solution for the Smart Product Pricing Challenge. It uses a state-of-the-art multi-modal ensemble model to predict product prices based on their images and text descriptions.

## Dataset 📦

This project uses a product pricing dataset (Amazon ML Challenge format) 
consisting of ~75,000 product listings for training and 75,000 for testing.
Each listing includes:

- **`catalog_content`** — the product's text description/title
- **A product image**, linked via `sample_id`
- **Derived numerical features** — pack size / item quantity and 
  weight or volume, automatically extracted from the product description 
  using regex parsing (e.g. "pack of 6", "500 ml", "2.5 lb")
- **Target variable** — the product's price

Of the 74,999 training samples, 15% was held out as a validation split 
to tune the ensemble blend weight and measure SMAPE.
## Results 📊

| Model | SMAPE (%) |
| **Final Blended Ensemble (LightGBM + XGBoost)** | **43.33%** |

Trained on 74,999 samples with a fused 2,306-dimension feature vector (vision + text + numerical), validated on a 15% held-out split.

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
2.  **XGBoost Regressor** 

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
