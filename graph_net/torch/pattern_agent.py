import importlib.util
import inspect
import os
from typing import List, Dict, Any

import torch
from torch.fx import GraphModule
from torch.fx.passes.shape_prop import ShapeProp


def _meta_to_tensor(meta_cls) -> torch.Tensor:
    """Convert a weight_meta class into a torch.Tensor.

    When meta_cls.data is None, generate a tensor using normal distribution
    with provided mean/std if available, otherwise zeros.
    """
    shape = getattr(meta_cls, "shape")
    dtype_str = getattr(meta_cls, "dtype")
    device = getattr(meta_cls, "device", "cpu")
    mean = getattr(meta_cls, "mean", 0.0)
    std = getattr(meta_cls, "std", 1.0)
    data = getattr(meta_cls, "data", None)

    dtype = eval(dtype_str) if isinstance(dtype_str, str) else torch.float32
    if data is not None:
        t = torch.tensor(data, dtype=dtype, device=device)
        if list(t.shape) != list(shape):
            t = t.view(*shape)
        return t
    # generate random tensor
    if std is None or std == 0:
        return torch.zeros(*shape, dtype=dtype, device=device)
    return torch.randn(*shape, dtype=dtype, device=device) * std + (mean or 0.0)


def load_model_and_inputs(sample_dir: str) -> (GraphModule, List[torch.Tensor]):
    """Load GraphModule and sample tensors from a GraphNet sample directory."""
    model_py = os.path.join(sample_dir, "model.py")
    weight_meta_py = os.path.join(sample_dir, "weight_meta.py")

    # load model module
    spec = importlib.util.spec_from_file_location("graph_module", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gm = mod.GraphModule()

    # load weight meta
    spec_w = importlib.util.spec_from_file_location("weight_meta", weight_meta_py)
    meta_mod = importlib.util.module_from_spec(spec_w)
    spec_w.loader.exec_module(meta_mod)

    sig = inspect.signature(gm.forward)
    tensors: List[torch.Tensor] = []
    for name in sig.parameters:
        cls_name = f"Program_weight_tensor_meta_{name}"
        meta_cls = getattr(meta_mod, cls_name)
        tensors.append(_meta_to_tensor(meta_cls))
    return gm, tensors


def profile_top_nodes(gm: GraphModule, inputs: List[torch.Tensor], topk: int = 5):
    """Profile the graph and return top-k nodes sorted by estimated time."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
        gm(*inputs)
    op_to_time: Dict[str, float] = {}
    for evt in prof.key_averages():
        if evt.key.startswith("aten::"):
            op_to_time[evt.key] = evt.self_cpu_time_total / max(evt.count, 1)

    node_times: Dict[Any, float] = {}
    for node in gm.graph.nodes:
        if node.op == "call_function":
            key = str(node.target).replace("<built-in function ", "").replace(">", "")
            key = key.replace("torch.ops.", "")
            key = key.replace("::", "::")
            for op_name, t in op_to_time.items():
                # match using the base operator name
                if op_name.split("::")[-1] in key:
                    node_times[node] = t
                    break
    top_nodes = sorted(node_times.items(), key=lambda kv: kv[1], reverse=True)[:topk]
    return [n for n, _ in top_nodes]


def expand_khop(gm: GraphModule, seed_nodes: List[Any], k: int = 2) -> GraphModule:
    cand = set(seed_nodes)
    frontier = set(seed_nodes)
    for _ in range(k):
        nxt = set()
        for n in frontier:
            nxt.update(n.all_input_nodes)
            nxt.update(n.users.keys())
        frontier = nxt - cand
        cand.update(nxt)
    new_graph = torch.fx.Graph()
    node_map: Dict[Any, torch.fx.Node] = {}
    for node in gm.graph.nodes:
        if node in cand:
            node_map[node] = new_graph.node_copy(node, lambda x: node_map.get(x))
    subgm = torch.fx.GraphModule(gm, new_graph)
    return subgm


def encode_subgraph(gm: GraphModule) -> Dict[str, Any]:
    ops = []
    edges = []
    metas = []
    node_to_idx = {}
    for idx, node in enumerate(gm.graph.nodes):
        ops.append(str(node.target))
        node_to_idx[node] = idx
        tm = node.meta.get("tensor_meta")
        if tm is not None:
            metas.append({"shape": list(tm.shape), "dtype": str(tm.dtype)})
        else:
            metas.append({"shape": None, "dtype": None})
    for node in gm.graph.nodes:
        dst = node_to_idx[node]
        for src_node in node.all_input_nodes:
            if src_node in node_to_idx:
                src = node_to_idx[src_node]
                edges.append((src, dst))
    return {"ops": ops, "edges": edges, "metas": metas}


def collect_subgraph_dataset(sample_dir: str, topk_nodes: int = 5, khop: int = 2):
    gm, tensors = load_model_and_inputs(sample_dir)
    # annotate shapes
    ShapeProp(gm).propagate(*tensors)
    hot_nodes = profile_top_nodes(gm, tensors, topk=topk_nodes)
    dataset = []
    for node in hot_nodes:
        sub = expand_khop(gm, [node], k=khop)
        enc = encode_subgraph(sub)
        dataset.append(enc)
    return dataset


class GraphSAGEDataset(torch.utils.data.Dataset):
    """Torch Dataset wrapping encoded subgraphs for GraphSAGE training."""

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        ops = {op for d in data for op in d["ops"]}
        self.op_to_idx = {op: i for i, op in enumerate(sorted(ops))}
        self.num_ops = len(self.op_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        op_idx = torch.tensor([self.op_to_idx[o] for o in item["ops"]], dtype=torch.long)
        if item["edges"]:
            edge_index = torch.tensor(item["edges"], dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
        label = torch.tensor(0)
        return op_idx, edge_index, label


class SimpleGraphSAGE(torch.nn.Module):
    """Minimal GraphSAGE-like model operating on op indices."""

    def __init__(self, num_ops: int, hidden_dim: int = 32):
        super().__init__()
        self.embed = torch.nn.Embedding(num_ops, hidden_dim)
        self.lin_self = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin_neigh = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, 2)

    def forward(self, op_idx: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.embed(op_idx)
        if edge_index.numel() == 0:
            neigh = torch.zeros_like(x)
        else:
            src, dst = edge_index
            neigh = torch.zeros_like(x)
            neigh.index_add_(0, dst, x[src])
            deg = torch.bincount(dst, minlength=x.size(0)).clamp(min=1).unsqueeze(1)
            neigh = neigh / deg
        h = self.lin_self(x) + self.lin_neigh(neigh)
        h = torch.relu(h)
        g = h.mean(dim=0)
        return self.out(g)


def train_graphsage(
    dataset: GraphSAGEDataset, epochs: int = 5, lr: float = 1e-3
) -> SimpleGraphSAGE:
    """Train a simple GraphSAGE model on the dataset."""

    model = SimpleGraphSAGE(dataset.num_ops)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for _ in range(epochs):
        for op_idx, edge_index, label in loader:
            opt.zero_grad()
            logits = model(op_idx[0], edge_index[0])
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), label)
            loss.backward()
            opt.step()
    return model


def build_dataset_from_samples(
    samples_root: str, topk_nodes: int = 5, khop: int = 2
) -> GraphSAGEDataset:
    """Collect subgraphs from all sample directories under ``samples_root``."""

    all_data: List[Dict[str, Any]] = []
    for name in os.listdir(samples_root):
        sample_dir = os.path.join(samples_root, name)
        if os.path.isdir(sample_dir):
            all_data.extend(
                collect_subgraph_dataset(sample_dir, topk_nodes=topk_nodes, khop=khop)
            )
    return GraphSAGEDataset(all_data)
