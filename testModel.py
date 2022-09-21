import torch
from PIL import Image
from torchvision import datasets, transforms

# test model

# load images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
images, labels = next(iter(valloader))

# load model
model = torch.load('./model.pt')
model.eval()

# check my image
img = Image.open("testImages/img.png")

img_tensor = transform(img)

# 6
# plt.imshow(img)

with torch.no_grad():
    logps = model(img_tensor)
ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print(probab)

# check all MNIST
correct_count, all_count = 0, 0
for images, labels in valloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if true_label == pred_label:
            correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count)
print("Model Accuracy =", (correct_count / all_count))
