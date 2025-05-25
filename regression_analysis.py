import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("insurance.csv")  # CSVfile in same directory

def regression_analysis(x_col, y_col):
    X = df[[x_col]]
    y = df[y_col]
    X = sm.add_constant(X)  # Add intercept

    model = sm.OLS(y, X).fit()
    print(f"\nRegression summary: {y_col} ~ {x_col}")
    print(model.summary())

    # Plot
    plt.figure(figsize=(8, 6))
    sns.regplot(x=x_col, y=y_col, data=df, line_kws={"color": "red"})
    plt.title(f'Regression: {y_col} vs {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.show()

# Run analyses

regression_analysis('age', 'bmi')
regression_analysis('age', 'charges')
