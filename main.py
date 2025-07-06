import argparse
import json
from src.train_tf import train_tf_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    train_tf_model(args.model, config)

if __name__ == "__main__":
    main()
