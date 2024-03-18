import datasets
import pandas as pd

dataset = datasets.load_dataset("GeoffVdr/cv8_trainval_processed")

diabetes_data = pd.read_csv("diabetes.csv")
