# Datascience X Logistic Regression

![Python](https://img.shields.io/badge/python-3.11-blue)

## Project Structure

The project is organized as follows:
```
dslr/
├── dataset/               # Contains the training and testing datasets
│   ├── dataset_train.csv  # Training dataset
│   ├── dataset_test.csv   # Testing dataset
├── results/               # Stores model parameters and results
│   ├── model_params_*.txt # Parameters for each house
│   ├── houses.csv         # Output predictions
│   └── .not_delete        # Placeholder file
├── src/                   # Source code for the project
│   ├── utils.py           # Utility functions
│   ├── logreg_train.py    # Logistic regression training script
│   ├── logreg_predict.py  # Logistic regression prediction script
│   └── Visualization/     # Scripts for data visualization
│       ├── scatter_plot.py
│       ├── pair_plot.py
│       ├── histogram.py
│       └── describe.py
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

## Dataset

The dataset consists of two CSV files:
- `dataset_train.csv`: Contains the training data with features and labels.
- `dataset_test.csv`: Contains the testing data without labels.

Each row represents a student, and the columns include various features such as scores in different subjects, personal details, and the target label (Hogwarts House).

## Install Dependencies

On Linux and MacOS:

```sh
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the Program

### Train the Model
To train the logistic regression model:
```sh
python3 src/logreg_train.py
```

### Make Predictions
To make predictions using the trained model:
```sh
python3 src/logreg_predict.py
```

### Visualize Data
To generate visualizations:
- Scatter plot:
  ```sh
  python3 src/Visualization/scatter_plot.py
  ```
- Pair plot:
  ```sh
  python3 src/Visualization/pair_plot.py
  ```
- Histogram:
  ```sh
  python3 src/Visualization/histogram.py
  ```
- Describe dataset:
  ```sh
  python3 src/Visualization/describe.py
  ```
