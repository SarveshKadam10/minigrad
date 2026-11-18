from graphviz import Digraph


def trace(root):
    nodes = set()
    edges = set()

    def build(node):
        if node not in nodes:
            nodes.add(node)

            for child in node._prev:
                edges.add((child, node))
                build(child)

    build(root)

    return nodes, edges


def draw_graph(root, filename="computation_graph"):
    nodes, edges = trace(root)

    dot = Digraph(
        format="png",
        graph_attr={"rankdir": "LR"}
    )

    for node in nodes:
        uid = str(id(node))

        dot.node(
            uid,
            label=f"{node.label} | data={node.data:.4f} | grad={node.grad:.4f}",
            shape="record"
        )

        if node._op:
            op_id = uid + node._op

            dot.node(
                op_id,
                label=node._op
            )

            dot.edge(op_id, uid)

    for child, parent in edges:
        child_id = str(id(child))
        parent_id = str(id(parent))

        if parent._op:
            dot.edge(child_id, parent_id + parent._op)
        else:
            dot.edge(child_id, parent_id)

    dot.render(
        f"images/{filename}",
        cleanup=True
    )