import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from att_unet import AttU_Net
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 判断知否存在gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image(img_path, label_path):
    img_x = Image.open(img_path)
    img_y = Image.open(label_path)

    data_transform = transforms.Compose([
        # transforms.Resize((256,256),interpolation=Image.BILINEAR),
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])
    target_transform = transforms.Compose([
        # transforms.Resize((256,256),interpolation=Image.BILINEAR),
        transforms.ToTensor()
    ])

    img_x = data_transform(img_x)
    img_y = target_transform(img_y)

    return img_x, img_y


def predict(num_classes, input_channels, img_path, label_path):
    print("=> creating model Unet")
    model = AttU_Net(input_channels, num_classes)
    model = model.to(device)
    model.eval()
    model_path = "./models/liverdataset_Unet_TRAIN/model_87.pth"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    img_x, img_y = get_image(img_path, label_path)

    img_x = img_x.to(device)
    img_y = img_y.to(device)

    output = model(img_x.unsqueeze(0))

    out = F.log_softmax(output, dim=1)
    pre_label = out.max(1)[1].squeeze().cpu().data.numpy()

    gt_label = img_y.squeeze().cpu().data.numpy()

    plt.subplot(1, 2, 1)
    plt.title("label_img")
    plt.imshow(gt_label, cmap=plt.cm.gray)
    plt.subplot(1, 2, 2)
    plt.title("predict_img")
    plt.imshow(pre_label, cmap=plt.cm.gray)
    plt.pause(0.1)
    plt.show()


if __name__ == '__main__':
    img_path = "./data/val/398.png"
    label_path = "./data/val_label/398_mask.png"
    predict(num_classes=2, input_channels=3, img_path=img_path, label_path=label_path)
