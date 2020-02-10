from classification import Dataset, LeafNode, DecisionTreeClassifier
from collections import deque
from eval import Evaluator

class DecisionTreePruner(object):

    def __init__(self, tree, valid):
        self.tree = tree
        self.valid = valid
        self.eval = Evaluator()

    def _prune_children(self, parent, node, direction):
        left_pred, right_pred = node.left.prediction, node.right.prediction
        left_count = parent.counts[left_pred]
        right_count = parent.counts[right_pred]

        new_pred = left_pred if left_count > right_count else right_pred

        if direction == 'L':
            parent.left = LeafNode(str(new_pred), prediction=new_pred)
        else:
            parent.right = LeafNode(str(new_pred), prediction=new_pred)

    def _unprune_children(self, parent, node, direction):
        if direction == 'L':
            parent.left = node
        else:
            parent.right = node

    def _get_accuracy(self):
        preds = self.tree.predict(valid.features)
        confusion = self.eval.confusion_matrix(preds, valid.labels)
        accuracy = self.eval.accuracy(confusion)

        return accuracy

    def _find_prunable_nodes(self):

        def prunable(node):
            return type(node.left) is LeafNode and type(node.right) is LeafNode

        queue = deque([self.tree.root])
        prunable_nodes = []
        while queue:
            node = queue.popleft()
            if type(node) is not LeafNode:
                for i, child in enumerate([node.left, node.right]):
                    if prunable(child):
                        direction = 'L' if i == 0 else 'R'
                        prunable_nodes.append((node, child, direction))
                    else:
                        queue.append(child)
        return prunable_nodes

    def prune_tree(self):
        unpruned_accuracy = self._get_accuracy()
        # self.tree.serialise_model("pruned_model.pickle")
        improved = True

        while improved:
            max_pruned_accuracy = 0
            prunable_nodes = self._find_prunable_nodes()
            best = None

            for parent, node, direction in prunable_nodes:
                self._prune_children(parent, node, direction)
                pruned_accuracy = self._get_accuracy()

                if pruned_accuracy > max_pruned_accuracy:
                    max_pruned_accuracy = pruned_accuracy
                    best = (parent, node, direction)

                self._unprune_children(parent, node, direction)

            improved = max_pruned_accuracy > unpruned_accuracy

            if improved:
                parent, best_node, direction = best
                # print(unpruned_accuracy, max_pruned_accuracy)
                unpruned_accuracy = max_pruned_accuracy

                # print("Pruning node:", best_node, "with children", best_node.left, best_node.right)
                # print("")
                print("pruned")

                self._prune_children(parent, best_node, direction)

                # print(self.tree)

def setup(model, valid, test):
    pruner = DecisionTreePruner(model, valid)
    old = str(pruner.tree)

    preds = pruner.tree.predict(test.features)
    confusion = eval.confusion_matrix(preds, test.labels)
    old_accuracy = eval.accuracy(confusion)

    pruner.prune_tree()

    preds = pruner.tree.predict(test.features)
    confusion = eval.confusion_matrix(preds, test.labels)
    new_accuracy = eval.accuracy(confusion)

    print(old == str(pruner.tree))

    return old_accuracy, new_accuracy


if __name__ == "__main__":
    headers = ["x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar",
               "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]

    valid = Dataset("data/validation.txt")
    test = Dataset("data/test.txt")
    eval = Evaluator()

    full_model = DecisionTreeClassifier(headers)
    full_model.deserialise_model("data/model.pickle")

    noisy_model = DecisionTreeClassifier(headers)
    noisy_model.deserialise_model("data/model_noisy.pickle")

    print(setup(full_model, valid, test))
    print(setup(noisy_model, valid, test))
    # count = 0
    # for i, pred in enumerate(preds):
    #     # print(pred, valid.labels[i])
    #     if pred == valid.labels[i]:
    #         count += 1
    # acc = float(count) / float(len(valid.labels)) * 100
    # print("ACCURACY = ", acc)
