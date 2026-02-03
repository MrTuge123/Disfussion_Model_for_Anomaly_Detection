from torchvision import datasets, transforms

# Define a transform to convert images to tensors
transform = transforms.ToTensor()

# Download and load training data
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Download and load test data
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Access a single example
image, label = train_dataset[0]
print(image.shape, label)
