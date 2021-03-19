import argparse
from PIL import Image
from model import *
import matplotlib.pyplot as plt

# arguments to command line
parser = argparse.ArgumentParser(description="Train model")
parser.add_argument("checkpoint", default=[None, None], nargs='+', help="checkpoint to load model")
parser.add_argument("image_path", type=str, help="file path of image to predict")

# get arugments
args = parser.parse_args()
checkpoint = args.checkpoint
image_path = args.image_path

# set cpu
device = torch.device("cpu")

# load model
model = TwoBinaryClassifiers().to(device)
if checkpoint[0]:
    model.bc1 = load_model("binary_classifier_1", checkpoint[0])
if checkpoint[1]:
    model.bc2 = load_model("binary_classifier_2", checkpoint[1])

# display image and its predicted label
with open(image_path, 'rb') as f:
    image = Image.open(f)
    prediction = predict(model, image)
    plt.imshow(image)
    plt.title("Predicted label: {}".format(prediction))
    plt.show()
    f.close()
