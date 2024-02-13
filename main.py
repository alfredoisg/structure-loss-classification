from lightning_modules.lightning_modules import LitLeNet5

model = LitLeNet5(num_classes=3, learning_rate=0.001)

print(model)
