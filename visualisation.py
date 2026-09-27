import seaborn as sns
import matplotlib.pyplot as plt

def plot_departure_reasons(df):
    plt.figure(figsize=(10,6))
    sns.countplot(x="JobRole", hue="Attrition", data=df)
    plt.xticks(rotation=45)
    plt.title("Attrition by Job Role")
    plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap")
    plt.show()