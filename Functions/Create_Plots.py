import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def create_plots(results: list[list], data_drifts: list[list], concept_drifts: list[list]):
    """
        Creates the plots of the metric we choose for our model

        Args:
            results: List with the metrics (the mean metric[1][i] and the windowing metric[0][i] of every model wee tested)
            data_drifts: Data drift points for each model
            concept_drifts: Concept drifts points for each model

        returns: Nothing

    """

    fig = make_subplots(rows=len(results[0]), cols=1,
                        subplot_titles=[f"Plot for pipeline {i+1}" for i in range(len(results[0]))])

    for i in range(len(results[0])):

        # plot
        fig.add_trace(go.Scatter(x=np.arange(1, len(results[0][i]) + 1), y=results[0][i], mode='lines',
                                 name=f'Metric for pipeline{i+1}'), row=i + 1, col=1)

        fig.add_trace(go.Scatter(x=np.arange(1, len(results[1][i]) + 1), y=results[1][i], mode='lines',
                                 name=f'Average Metric for pipeline{i + 1}'), row=i + 1, col=1)

        # add data drifts in plot
        for drift_index in data_drifts[i]:
            fig.add_vline(x=drift_index, line=dict(color='green', dash='dash'),
                          name='Data Drift Detected', row=i + 1, col=1)

        # add concept drifts in plot
        for drift_index in concept_drifts[i]:
            fig.add_vline(x=drift_index, line=dict(color='red', dash='dash'),
                          name='Concept Drift Detected', row=i + 1, col=1)


        fig.update_xaxes(title_text="Sample number", row=i + 1, col=1)
        fig.update_yaxes(title_text="Metric", range=[0, 1], row=i + 1, col=1)

    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='green', dash='dash'),
                             name='Data Drift'))

    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='red', dash='dash'),
                             name='Concept Drift'))

    fig.update_layout(
        height=340 * len(results[0]),
        title='Metric plot',
        legend_title='Legend',
        hovermode='x'
    )
    fig.show()
