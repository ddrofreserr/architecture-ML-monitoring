import argparse
from data.data_loader import load_data
from src.train import train_model
from src.predict import predict

def main(source="synthetic", mode="train", retrain=False):
    data = load_data(source)

    if mode == "train":
        train_model(data, retrain=retrain)
    elif mode == "predict":
        predictions = predict(data)
        print(predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--source", default="synthetic", choices=["synthetic", "db"])
    parser.add_argument("--mode", default="train", choices=["train", "predict"])
    parser.add_argument("--retrain", action="store_true") 
    args = parser.parse_args()
    # main(source=args.source, mode=args.mode, retrain=args.retrain)
    main(source='db', mode=args.mode)
