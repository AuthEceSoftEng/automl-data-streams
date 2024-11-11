from matplotlib import pyplot as plt
import pandas as pd


def data_plot(data: list[dict], step: int):
    """
        Creates the plots of the dataset

        Args:
            data: list with dataset
            step: step of the visualization of the dataset

        returns: Nothing

    """

    df = pd.DataFrame(data)  # make the datset a pandas dataframe

    df = df.iloc[::step, :]  # Plot every 50 data points - for visualizations reasons

    df.plot(subplots=True)

    plt.show()
