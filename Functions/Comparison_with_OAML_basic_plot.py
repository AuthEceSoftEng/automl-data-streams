import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def compare_with_oaml(results: list[list]):
    """
            Creates the plots of the metric we choose to compare some methods with OAML

            Args:
                results: List with the metrics results(the last one list contains the metric results for OAML with step 1000 and start in 6000)

            returns: Nothing

        """

    # Create an empty figure
    fig = go.Figure()

    for i in range(len(results)):
        # the last is the result of the oaml
        if i != len(results)-1:
            fig.add_trace(go.Scatter(x=np.arange(1, len(results[i]) + 1), y=results[i], mode='lines', name=f'Average metric for pipeline{i+1}'))
        else:
            fig.add_trace(go.Scatter(x=np.arange(6000, (len(results[i])+5) * 1000 + 1, 1000), y=results[i], mode='lines', name=f'Average metric for OAML-basic'))

    # Update layout for titles and axis labels
    fig.update_layout(
        title="Metric plot",
        xaxis_title="Sample number",
        yaxis_title="Metric",
        showlegend=True  # Display the legend
    )

    # Show the figure
    fig.show()
