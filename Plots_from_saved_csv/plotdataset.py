import os
import pandas as pd
import matplotlib.pyplot as plt
from LoanDataset.loandataset_2_class import LoanDataset

imagespath = r"SET_THIS"

## CREATE DATASET
# Number of data instances
numclasses = 2
datalimit = 20000
conceptdriftpoints={2500: "growth", 4000: "crisis",
                                                  6000: "growth", 10000: "normal",
                                                  13000: "growth", 16000: "normal",
                                                  18000: "crisis"}
datadriftpoints={2000: "crisis", 6000: "growth",
                                               8000: "normal", 9000: "crisis",
                                               12000: "growth", 14000: "crisis",
                                               16000: "normal"}

# Load loan dataset
dataset = LoanDataset(seed=42)

# Have a look at the data (note that data are a stream, so this "view"
# should not be given to the algorithms, it is only for intuition purposes)
data = []
for i, (x, y) in enumerate(dataset.take(datalimit)):
    if i in conceptdriftpoints:
        dataset.generate_concept_drift(conceptdriftpoints[i])
    if i in datadriftpoints:
        dataset.generate_data_drift(datadriftpoints[i])
    data.append({**x, "y": y})
    #data.append({**x, "y": 1 if y else 0})
df = pd.DataFrame.from_dict(data)
df = df.iloc[::250, :]#df.sample(n = 300, random_state=42).sort_index()
axes = df.plot(subplots=True, figsize=(4.5, 6), color="#1f77b4", legend=False)
for c, ax in enumerate(axes):
#    ax.axvline(400, color='red', linestyle='dashed')
#    ax.legend(loc='center left')

    for i in conceptdriftpoints:
        cd = ax.axvline(i, ymin = -0.1, ymax = 1.1, clip_on=False, zorder=10, label = "Concept drifts", color = "gray", linestyle = "dashed")
    for i in datadriftpoints:
        dd = ax.axvline(i, ymin = -0.1, ymax = 1.1, clip_on=False, zorder=10, label = "Data drifts", color = "gray", linestyle = "dotted", linewidth=1.75)
    ax.set_ylabel(df.columns[c] if df.columns[c] != "educationlevel" else "education", rotation=0, horizontalalignment='right', verticalalignment='center')
    ax.get_yaxis().set_label_coords(-0.01,0.5)
    ax.ticklabel_format(useOffset=False, style='plain')
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_locator(ticker.NullLocator())
    continue
    #ax.set_xlim(-2500, 20000)
#    continue
    text = ax.annotate(
        df.columns[c],
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif', zorder=20,
        bbox=dict(facecolor='0.9', edgecolor='none', pad=3.0, alpha=0.8))
#    text.set_alpha(0.4)
axes[0].legend([cd, dd],['Concept drifts', 'Data drifts'], loc = 'upper center', bbox_to_anchor=(0.5, 1.8), ncol=2)
#plt.tight_layout()
axes[-1].set_xlabel("Data instances")
axes[-1].set_xticks([0, 5000, 10000, 15000, 20000], [0, 5000, 10000, 15000, 20000])
#    ax.xaxis.set_major_locator(ticker.NullLocator())
plt.subplots_adjust(0.2, 0.1, 0.96, 0.93)
plt.savefig(os.path.join(imagespath, "loandataset.pdf"))
plt.show()
