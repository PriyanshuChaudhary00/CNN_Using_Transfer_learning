import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_Data(path):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(f"{path}/PetImages" , transform=transform)

    from torch.utils.data import random_split

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size]
    )

    train_loader = DataLoader(dataset=train_dataset , batch_size = 32 , shuffle=True)
    test_loader = DataLoader(dataset=test_dataset , batch_size = 32 , shuffle=True)
    return train_loader , test_loader