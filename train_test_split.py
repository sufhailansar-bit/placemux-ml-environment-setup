import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("sample_dataset.csv")

X = df[['Python','MachineLearning','DataAnalysis']]
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))
