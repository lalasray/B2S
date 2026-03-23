from b2s.config import DataConfig
from b2s.dataset import save_dataset


def main() -> None:
    """Generate the synthetic dataset splits used by the prototype."""
    config = DataConfig()
    # Save the default train/val/test splits to Code/data.
    paths = save_dataset(config, "Code/data")
    for split, path in paths.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
