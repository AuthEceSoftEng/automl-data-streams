import copy
from Functions.Sliding_Window_Class import SlidingWindow


def evaluation(y_real: list[list], y_predicted: list[list], metric_algorithm):
    """
        Create the evaluation (mean and windowing) for every model with the metric of our decision

        Args:
            y_real: list of the real target values that used in the model
            y_predicted: List with predicted values of every model we used
            metric_algorithm: The metric algorithm with which we evaluate the models

        returns:
            results_window: A list with the windowing metric values in each step for every model
            results_average: A list with the mean metric values in each step for every model

    """
    results_window = []
    results_average = []
    for i in range(len(y_real)):
        evaluate = []
        evaluate_average = []
        # create the instance of metric
        metric = copy.deepcopy(metric_algorithm)
        real_window = SlidingWindow(100)
        predict_window = SlidingWindow(100)
        metric_average = copy.deepcopy(metric_algorithm)
        for yr, yp in zip(y_real[i], y_predicted[i]):
            if yr is None and yp is None:
                evaluate.append(None)
                evaluate_average.append(None)
            else:
                real_window.add(yr)
                predict_window.add(yp)
                metric_average.update(yr, yp)
                evaluate_average.append(metric_average.get())

                metric = copy.deepcopy(metric_algorithm)
                for yreal, ypred in zip(real_window.get(), predict_window.get()):
                    metric.update(yreal, ypred)

                evaluate.append(metric.get())
        results_window.append(evaluate)
        results_average.append(evaluate_average)

    return results_window, results_average
