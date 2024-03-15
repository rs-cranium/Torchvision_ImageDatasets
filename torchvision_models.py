import torchvision.models as models

# Instantiate ResNet-18 model
model = models.resnet18(pretrained=True)

# Output the model architecture
print(model)
