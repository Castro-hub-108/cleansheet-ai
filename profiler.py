import matplotlib.pyplot as plt
import seaborn as sns

def generate_report(df):
    figs = []

    # Histogram
    fig1, ax1 = plt.subplots()
    df.hist(ax=ax1)
    figs.append(fig1)

    # Boxplot
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df.select_dtypes(include='number'), ax=ax2)
    figs.append(fig2)

    return figs
