# GI-GCN: Global Interacted Graph Convolutional Networks via Dominant Sets for Graph Classification

## abstract
Graph Convolutional Networks (GCNs) are defined based on aggregating the node information of adjacent nodes, which are usually treated as equally important, limiting the representational power of existing GCNs for graph classification. To address this shortcoming, we propose a novel Global Interacted Graph Convolutional Network (GI-GCN) that leverages the solution vectors maintained during the iterative updates of the Dominant Set to adaptively characterize the global importance distribution across all nodes. Specifically, at each convolution layer, this distribution is adopted to adaptively modulate the importance weights of node features before performing local message passing. We show that this convolution strategy can effectively capture highly correlated information between nonadjacent nodes through the Dominant Set algorithm, not only emphasizing critical graph-level information but also enhancing the discriminative power of graph representations. Furthermore, we optimize the memory complexity of the framework, significantly reducing the memory overhead associated with global interaction modeling. Experiments demonstrate the effectiveness of the proposed GI-GCN.
## Prerequisites

- pytorch 2.4.1+cu124
- torch_geometric 2.6.1

## Running the Project

1. **Run `main.py` for model training and testing:**
   ```bash
   python main.py
