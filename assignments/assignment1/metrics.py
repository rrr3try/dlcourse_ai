def binary_classification_metrics(prediction, ground_truth):
    """
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    """
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    num_of_samples = prediction.shape[0]
    comparison = [(prediction[i] == ground_truth[i], ground_truth[i])
                  for i in range(num_of_samples)]
    true_positive = len([c for c in comparison if c[0] and c[1]])
    false_positive = len([c for c in comparison if not c[0] and not c[1]])
    false_negative = len([c for c in comparison if not c[0] and c[1]])

    if true_positive == 0:
        precision = 0
        recall = 0
        accuracy = 0
        f1 = 0
    else:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        accuracy = (prediction == ground_truth).sum()/num_of_samples
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    positive = (prediction == ground_truth).sum()
    num_of_samples = prediction.shape[0]
    return positive/num_of_samples
