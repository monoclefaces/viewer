# Viewer

## Prerequisite

```
torch >= 1.4.0
torchvision >= 0.5.0
pyyaml >= 5.3

```

## Project Directory Tree

```bash
├── main.py
├── trainer
│   ├── __init__.py
│   ├── dataloader.py  # Dataset 및 DataLoader
│   └── trainer.py     # Trainer (train, valid 함수 등)
└── viewer
    ├── __init__.py
    ├── attention      # Attention Modules
    ├── evalutation    # Evaluation Methods
    └── saliency       # Attribution Method
```

