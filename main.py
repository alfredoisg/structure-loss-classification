from lightning_modules.lightning_modules import LitLeNet5
from train.train import get_features
from utils.utils import load_targets

model = LitLeNet5(num_classes=3, learning_rate=0.001, size_layer_1=10, size_layer_2=5)

targets = load_targets()
print(model)
