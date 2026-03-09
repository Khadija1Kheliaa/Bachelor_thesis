Wikipedia’s collaborative editing environment, while revolutionizing access to knowledge,
leaves the platform open to vandalism. Since existing detection methods rely mainly on edit
content or static user metadata, they often fail to capture the underlying behavioral patterns
of vandals. This thesis suggests a graph-based framework that integrates revision history
analysis with interaction graph construction to highlight the impact of network structure on
vandalism detection.

To validate this approach, a data-processing pipeline was developed to extract the revision
histories of 160 Wikipedia articles from 2011-2013, identify unique editors, and utilize ClueBot
NG reverts as a proxy for ground-truth labels. Through revision history analysis, individual
user features were extracted and user interactions were mapped into a graph, with edge types
inferred from computing the differences between revisions. A GraphSAGE model was then
implemented by combining these features with the underlying graph structure to classify
editors as either vandals or non-vandals.

The model was developed through three optimization phases to refine parameters and
determine the optimal configuration, ultimately achieving significant performance with an
Area Under the Curve (AUC) of 0.94. To provide an operational framework for real-world
deployment, a threshold analysis was performed to offer a control mechanism for balancing
the model’s sensitivity and precision. Finally, several ablation studies were conducted to
evaluate the system’s robustness: architectural ablation revealed graph connectivity to be a
fundamental pillar of the model’s effectiveness, while feature ablation highlighted the critical
importance of integrating network-based features.

Ultimately, these findings confirm that network structure serves as a critical dimension
in vandalism detection, uncovering behavioral patterns that content-centric methods fail to
capture.
