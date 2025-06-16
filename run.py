import argparse
from data.data_loader import load_data
from src.train import train_model
from src.predict import predict_with_explanation


def main(source="synthetic", mode="train", retrain=False, explain=True):

    if mode == "train":
        data = load_data(source)
        train_model(data, retrain=retrain)

    elif mode == "predict":
        data = load_data(source, drop_target=True)
        predictions = predict_with_explanation(data)
        print(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="synthetic", choices=["synthetic", "db", "external"])
    parser.add_argument("--mode", default="predict", choices=["train", "predict"])
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    main(source=args.source, mode=args.mode, retrain=args.retrain)
    # main(source='db', mode=args.mode)
