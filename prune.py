from classification import Dataset, LeafNode, DecisionTreeClassifier
from collections import deque
from eval import Evaluator

class DecisionTreePruner(object):

    def __init__(self, tree, labels, counts, valid_features, valid_labels):
        self.tree = tree
        self.label_count = {l: c for l, c in zip(labels, counts)}
        self.valid_features = valid_features
        self.valid_labels = valid_labels
        self.eval = Evaluator()

    def _prune_children(self, node):
        left_pred, right_pred = node.left.prediction, node.right.prediction
        left_count = self.label_count[left_pred]
        right_count = self.label_count[right_pred]

        if left_count > right_count:
            node.left = node.right = LeafNode(prediction=left_pred)
        else:
            node.left = node.right = LeafNode(prediction=right_pred)

    def _get_accuracy(self):
        preds = self.tree.predict(self.valid_features)
        confusion = self.eval.confusion_matrix(preds, self.valid_labels)
        accuracy = self.eval.accuracy(confusion)

        return accuracy

    def _find_prunable_nodes(self):
        queue = deque([self.tree.root])
        prunable_nodes = []
        while queue:
            node = queue.popleft()
            if type(node) is not LeafNode:
                prunable = type(node.left) is LeafNode and type(node.right) is LeafNode
                if prunable and node.left.prediction != node.right.prediction:
                    prunable_nodes.append(node)
                queue.append(node.left)
                queue.append(node.right)
        return prunable_nodes

    def prune_tree(self):
        unpruned_accuracy = self._get_accuracy()
        self.tree.serialise_model("pruned_model.pickle")
        improved = True

        while improved:
            prunable_nodes = self._find_prunable_nodes()
            max_pruned_accuracy = 0
            best_node = None

            for node in prunable_nodes:
                self._prune_children(node)
                pruned_accuracy = self._get_accuracy()

                if pruned_accuracy > max_pruned_accuracy:
                    max_pruned_accuracy = pruned_accuracy
                    best_node = node

                self.tree.deserialise_model("pruned_model.pickle")

            improved = max_pruned_accuracy > unpruned_accuracy
            if improved:
                self._prune_children(best_node)
                self.tree.serialise_model("pruned_model.pickle")


if __name__ == "__main__":
    headers = ["x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar",
               "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]

    valid = Dataset("data/validation.txt")
    train = Dataset("data/train_sub.txt")

    labels, counts = train.label_distribution()
    model = DecisionTreeClassifier(headers)
    model.deserialise_model("data/model_sub.pickle")

    pruner = DecisionTreePruner(model, labels, counts, valid.features, valid.labels)
    pruner.prune_tree()
