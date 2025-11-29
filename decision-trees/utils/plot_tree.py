from graphviz import Digraph


def plot_tree(node, feature_names=None, class_names=None, parent=None, edge_label=""):
    """
    Visualize your custom DecisionTree using Graphviz.

    node: root node of the tree
    feature_names: list of feature names
    class_names: list of class names corresponding to labels
    parent: used internally for recursion
    edge_label: "Yes"/"No" for left/right branch
    """
    dot = Digraph()

    def add_nodes_edges(node, parent=None, edge_label=""):
        if node is None:
            return

        # Leaf node
        if node.is_leaf():
            if class_names is not None:
                label = f"Class={class_names[node.value]}"
            else:
                label = f"Class={node.value}"
            dot.node(str(id(node)), label)
        else:
            # Internal node
            feature = (
                feature_names[node.feature_index]
                if feature_names is not None
                else f"X{node.feature_index}"
            )
            label = f"{feature} <= {node.threshold}"
            dot.node(str(id(node)), label)

            # Recursively add children
            add_nodes_edges(node.left, node, "Yes")
            add_nodes_edges(node.right, node, "No")

        # Add edge from parent to this node
        if parent is not None:
            dot.edge(str(id(parent)), str(id(node)), label=edge_label)

    add_nodes_edges(node)
    return dot
