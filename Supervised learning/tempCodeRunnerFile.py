
import pandas as pd
df = pd.read_json("A:/machine learning/health problem prediction/model/health_problems.json")
print(df['data_columns'])

df = pd.read_json("A:/machine learning/health problem prediction/model/precaution.json")
print(df['data_columns'])

df = pd.read_json("A:/machine learning/health problem prediction/model/home_remedies.json")
print(df['data_columns'])