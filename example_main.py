##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 example execution code
# Prepared by: Josiah Wang
##############################################################################

import numpy as np

from classification import DecisionTreeClassifier
from eval import Evaluator

if __name__ == "__main__":
    print("Loading the training dataset...");
    x = np.array([
            [5,7,1],
            [4,6,2],
            [4,6,3],
            [1,3,1],
            [2,1,2],
            [5,2,6]
        ])

    y = np.array(["A", "A", "A", "C", "C", "C"])

    print("Training the decision tree...")
    classifier = DecisionTreeClassifier("")
    classifier = classifier.train(x, y)

    print("Loading the test set...")

    x_test = np.array([
            [1,6,3],
            [0,5,5],
            [1,5,0],
            [2,4,2]
        ])

    y_test = np.array(["A", "A", "C", "C"])

    predictions = classifier.predict(x_test)
    print("Predictions: {}".format(predictions))

    classes = ["A", "C"];

    print("Evaluating test predictions...")
    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(predictions, y_test)

    print("Confusion matrix:")
    print(confusion)

    accuracy = evaluator.accuracy(confusion)
    print()
    print("Accuracy: {}".format(accuracy))

    (p, macro_p) = evaluator.precision(confusion)
    (r, macro_r) = evaluator.recall(confusion)
    (f, macro_f) = evaluator.f1_score(confusion)

    print()
    print("Class: Precision, Recall, F1")
    for (i, (p1, r1, f1)) in enumerate(zip(p, r, f)):
        print("{}: {:.2f}, {:.2f}, {:.2f}".format(classes[i], p1, r1, f1));

    print()
    print("Macro-averaged Precision: {:.2f}".format(macro_p))
    print("Macro-averaged Recall: {:.2f}".format(macro_r))
    print("Macro-averaged F1: {:.2f}".format(macro_f))
