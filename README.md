# ML Project – Raw Material Stock Prediction

This project is part of the TDT4173 course – Modern Machine Learning in Practice.  
We use Kaggle for submissions, and this repository contains code, data processing, experiments, and the final report.

## Create Environment

```bash
conda env create -f environment.yml
conda activate ML-project
```

## Description of the files given to students

README.md - this file

data/kernel/receivals.csv - training data  
data/kernel/purchase_orders.csv - imporant data that contains whats ordered

data/extended/materials.csv - optional data related to the different materials  
data/extended/transportaton.csv - optional data related to transporation

data/prediction_mapping.csv - mapping used to generate submissions
data/sample_submission.csv - demo submission file for Kaggle, predicting all zeros

Dataset definitions and explanation.docx - a documents that gives more details about the dataset and column names  
Machine learning task for TDT4173.docx - brief introduction to the task
kaggle_metric.ipynb - the score function we use in the Kaggle competition


## Project Structure

```bash
ML-project/
│
├── data/                  
│   ├── extended           
│   └── kernel
│
├── notebooks/
│   ├── long_notebook.ipynb      # master log of all work, sequential, full story
│   ├── submissions/             # each submission = notebook + metadata
│   │   ├── sub_2025-09-20_xgb/
│   │   │   ├── notebook.ipynb
│   │   │   └── notes.txt        # description of changes + Kaggle score
│   │   ├── sub_2025-09-23_lstm/
│   │   │   ├── notebook.ipynb
│   │   │   └── notes.txt
│   │   └── ...
│   │
│   └── dev/                      # parallel development notebooks
│       ├── model_xgb_A.ipynb     # e.g. Martin working on XGBoost
│       ├── model_lstm_B.ipynb    # e.g. Alma working on LSTM
│       └── ...
│
├── production/              # clean notebooks for final submission
│   ├── short_notebook_1.ipynb
│   ├── short_notebook_2.ipynb
│   └── report.ipynb or report.pdf
│
├── submissions/             # Kaggle submission CSVs (mirrors notebooks/submissions)
│   ├── sub_2025-09-20_xgb.csv
│   ├── sub_2025-09-23_lstm.csv
│   └── ...
│
├── environment.yml          # conda environment spec
├── README.md                # setup + usage guide
└── .gitignore               
```
