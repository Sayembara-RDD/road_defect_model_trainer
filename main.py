from ultralytics import YOLO
import argparse
import os


def main(args):
    epochs = int(args.epochs)
    train_yaml = args.train_yaml
    model_name = args.model_name
    batch = int(args.batch)
    pretrained_model = args.pretrained_model

    model = YOLO(pretrained_model)
    if not os.path.isfile(train_yaml):
        print("Train yaml is not found, Please input the correct file!")
        return
    model.train(data=train_yaml, epochs=epochs, name=model_name, batch=batch)

    metrics = model.val()

    # export model to onnx
    model.export(format="onnx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Arguments to train yolov8 model')
    parser.add_argument(
        '--train_yaml', help="path to yaml file to train model", default="example_train.yaml"),
    parser.add_argument(
        '--epochs', help="Epochs value to train the model", default=50)
    parser.add_argument(
        '--batch', help="Batch value to train the model", default=4)
    parser.add_argument('--model_name', help="Train model name")
    parser.add_argument('--pretrained_model',
                        help="Pretrained model to use", default="yolov8m.pt")
    args = parser.parse_args()

    main(args)
