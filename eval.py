##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks:
# Complete the following methods of Evaluator:
# - confusion_matrix()
# - accuracy()
# - precision()
# - recall()
# - f1_score()
##############################################################################

import numpy as np
import classification as cl

class Evaluator(object):
    """ Class to perform evaluation
    """

    def confusion_matrix(self, prediction, annotation, class_labels=None):
        """ Computes the confusion matrix.

        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.

        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """

        if not class_labels:
            class_labels = np.unique(annotation)
        class_map = {cl: i for i, cl in enumerate(class_labels)}

        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)
        for a, p in zip(annotation, prediction):
            a, p = class_map[a], class_map[p]
            confusion[a][p] += 1

        return confusion


    def accuracy(self, confusion):
        """ Computes the accuracy given a confusion matrix.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions

        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """

        diag_sum = confusion.trace()
        accuracy = diag_sum / confusion.sum()

        return accuracy


    def precision(self, confusion):
        """ Computes the precision score per class given a confusion matrix.

        Also returns the macro-averaged precision across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the precision score for each
            class in the same order as given in the confusion matrix.
        float
            The macro-averaged precision score across C classes.
        """

        # Initialise array to store precision for C classes
        p = np.zeros((len(confusion), ))
        c,_ = confusion.shape
        macro_p = 0

        for cl in range(c):
            col = confusion[:, cl]
            p[cl] = confusion[cl, cl] / col.sum()

            macro_p += p[cl]

        macro_p /= c

        return (p, macro_p)


    def recall(self, confusion):
        """ Computes the recall score per class given a confusion matrix.

        Also returns the macro-averaged recall across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the recall score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged recall score across C classes.
        """

        # Initialise array to store recall for C classes
        r = np.zeros((len(confusion), ))
        c,_ = confusion.shape
        macro_r = 0

        for cl in range(c):
            row = confusion[cl, :]
            r[cl] = confusion[cl, cl] / row.sum()

            macro_r += r[cl]

        macro_r /= c

        return (r, macro_r)


    def f1_score(self, confusion):
        """ Computes the f1 score per class given a confusion matrix.

        Also returns the macro-averaged f1-score across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the f1 score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged f1 score across C classes.
        """

        # Initialise array to store recall for C classes
        f = np.zeros((len(confusion), ))
        c, _ = confusion.shape

        p, macro_p = self.precision(confusion)
        r, macro_r = self.recall(confusion)

        for cl in range(c):
            f[cl] = (2 * p[cl] * r[cl]) / (p[cl] + r[cl])

        macro_f = (2 * macro_p * macro_r) / (macro_p + macro_r)

        return (f, macro_f)

    def k_split(self, x, y, k):
        data = np.hstack((x, np.array([y]).T))
        np.random.shuffle(data)
        return np.array_split(data, k)

    def k_fold_cv(self, x, y, k, scoring):

        k = len(y) if k > len(y) else k

        if k < 2:
            print("k must be at least 2 as the minimum number of splits is 2")
            exit(0)

        splits = self.k_split(x, y, k)
        test_set, train_set = np.array([]), np.array([])
        scores = np.array([])
        model = cl.DecisionTreeClassifier()

        for idx, fold in enumerate(splits):

            x_test = fold.T[:-1]
            y_test = fold.T[-1]

            for i in range(len(splits)):
                if i != idx:
                    if len(train_set) > 0:
                        train_set = np.vstack((train_set, splits[i]))
                    else:
                        train_set = splits[i]

            x_train = train_set.T[:-1]
            y_train = train_set.T[-1]

            model.train(x_train.T, y_train)
            conf = self.confusion_matrix(model.predict(x_test.T), y_test)
            avg = None

            if scoring == "accuracy":
                avg = self.accuracy(conf)
            elif scoring == "precision":
                _, avg = self.precision(conf)
            elif scoring == "recall":
                _, avg = self.recall(conf)
            elif scoring == "f1":
                _, avg = self.f1_score(conf)
            else:
                print("Invalid scoring metric. Please enter accuracy, precision, recall or f1")
                exit(0)

            scores = np.append(scores, avg)

        return np.mean(scores), np.std(scores)

if __name__ == "__main__":
    ds = cl.Dataset()
    ds.read("data/toy.txt")

    eval = Evaluator()
    print(eval.k_fold_cv(ds.features, ds.labels, 4, "recall"))
