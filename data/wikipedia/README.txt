Wikipedia Article Networks

Description

The data was collected from the English Wikipedia (December 2018). These datasets represent page-page networks on specific topics (chameleons, crocodiles and squirrels). Nodes represent articles and edges are mutual links between them. The edges csv files contain the edges - nodes are indexed from 0. The features json files contain the features of articles - each key is a page id, and node features are given as lists. The presence of a feature in the feature list means that an informative noun appeared in the text of the Wikipedia article. The target csv contains the node identifiers and the average monthly traffic between October 2017 and November 2018 for each page.  For each page-page network we listed the number of nodes an edges with some other descriptive statistics.

- Directed: No.
- Node features: Yes.
- Edge features: No.
- Node labels: Yes. Continuous target.
- Temporal: No.

|   | Chameleon  | Crocodile  | Squirrel  |
|---|---|---|---|
| Nodes |2,277   | 11,631  |  5,201 |
| Edges | 31,421  |170,918 |  198,493 |
| Density |  0.012 | 0.003  | 0.015 |
| Transitvity | 0.314| 0.026 | 0.348 |

Possible Tasks

- Regression
- Link prediction
- Community detection
- Network visualization
