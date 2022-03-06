# Explanations about Data

## Data Format

The data format is largely following the [Graph Transformer](https://github.com/Amazing-J/structural-transformer) and our another [work](https://github.com/muyeby/AMR-Backparsing/) on AMR-to-Text generation.

Assuming that an input AMR graph has N nodes after bpe (we add a dummy root node to the graph) and the corresponding text has M tokens after bpe, then

1. Each line of 'dev.concept' includes N terms, storing the concepts of the AMR graph.

2. Each line of 'dev.path' includes N*N terms, storing the "relation" matrix of the AMR graph. For example, the "relation" (or path) between node_{i} and node_{j} are stored at the ixj th term. The "relation" here refers to the shortest path in AMR graph between node_{i} and node_{j}.

3. Each line of 'dev.rel' includes M*M terms, storing the relation matrix of the sentence. For example, the relation between word_{i} and word_{j} are stored at the i x j th term.

4. Each line of 'dev.mask' includes M*M terms, storing the relation mask matrix of the sentence. The relation mask value is 1 when there is a relation between word_{i} and word_{j}, 0 otherwise.

## Data Usage

1. The hier model takes a _projected_ AMR graph as input, thus 'dev.concept' and 'dev.path' are not used.

2. The dual model processes an AMR graph independently, thus 'dev.rel' and 'dev.mask' are not used.
