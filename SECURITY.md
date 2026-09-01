# Security policy

Please report suspected credential exposure, unsafe checkpoint loading, or
path-handling vulnerabilities privately to the repository maintainers before
opening a public issue.

NREM processes local NumPy arrays and PyTorch checkpoints. Only load datasets
and checkpoints from trusted sources. The project does not require network
access during training or evaluation.
