from river.metrics import Accuracy


def accuracy_check(mean_accuracy, y_true_buffer, y_predicted_buffer, threshold: float):
    """
        Compare the accuracy of a model in a window with the mean accuracy
        of the model and decide whether to find for a better or not

        Args:
            mean_accuracy: The mean accuracy until the time in the decision
            y_true_buffer: The true values of the window
            y_predicted_buffer: The predicted values in the window
            threshold: The number that is the minimum value of the difference in the accuracy

        returns:
            need_change: A boolean that return False if the model don needs update, and 1 if it needs update
    """
    need_change = False

    accuracy = Accuracy()

    for y_true, y_predicted in zip(y_true_buffer, y_predicted_buffer):
        accuracy.update(y_true, y_predicted)

    if mean_accuracy.get() - accuracy.get() >= threshold:
        need_change = True

    return need_change
