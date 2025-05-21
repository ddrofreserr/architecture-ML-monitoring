import argparse
from data.data_loader import load_data
from src.train import train_model
from src.predict import predict, predict_with_explanation


def main(source="synthetic", mode="train", retrain=False, explain=True):
    data = load_data(source)

    if mode == "train":
        train_model(data, retrain=retrain)

    elif mode == "predict":
        if explain:
            predictions = predict_with_explanation(data)
        # else:
        #     predictions = predict(data)
        print(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="synthetic", choices=["synthetic", "db", "external"])
    parser.add_argument("--mode", default="predict", choices=["train", "predict"])
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.set_defaults(explain=True)
    args = parser.parse_args()
    main(source=args.source, mode=args.mode, retrain=args.retrain, explain=args.explain)
    # main(source='db', mode=args.mode)
