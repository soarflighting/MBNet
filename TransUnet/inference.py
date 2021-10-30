import torch
from metrics import calc_iou,eval_seg
from networks.vit_seg_modeling import VisionTransformer,CONFIGS

import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image as Image
import numpy as np
import pydensecrf.densecrf as dcrf
from cv2 import imread, imwrite,resize
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import cv2

# 判断知否存在gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_image(img_path,label_path):
    # print(img_path)
    img_x = Image.open(img_path)
    img_y = Image.open(label_path).convert('L')

    data_transform = transforms.Compose([
                            transforms.Resize((256, 256), interpolation=Image.BILINEAR),
                            transforms.ToTensor(),  # -> [0,1]
                            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
        ])
    target_transform = transforms.Compose([
                transforms.Resize((256,256),interpolation=Image.BILINEAR),
                transforms.ToTensor()
    ])

    img_x = data_transform(img_x)
    img_y = target_transform(img_y)

    return img_x,img_y





def predict(img_path,label_path,in_channels,num_classses):

    print("=====================creating Model BANet==================")
    config_vit = CONFIGS['R50-ViT-B_16']
    config_vit.n_classes = 2
    config_vit.n_skip = 3
    model = VisionTransformer(config_vit, 256, 2)
    model.eval()
    model = model.to(device)
    # model_path = "./models/dataset03_BANet_TRAIN/model_192.pth"
    # model_path = "./models/dataset03_p_BANet_TRAIN/model_298.pth"
    # model_path = "./models/dataset08_BANet_TRAIN/model_208.pth"
    # model_path = "./models/dataset10_BANet_TRAIN/model_278.pth"
    # model_path = "./models/dataset_resnet4_BANet_TRAIN/model_197.pth"
    # model_path = "./models/liver_data_TransUnet_TRAIN/model_272.pth"

    model_path = "./models/kvasir_data_TransUnet_TRAIN/model_192.pth"

    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    img,label = get_image(img_path,label_path)

    img = img.to(device)
    label = label.to(device)

    output = model(img.unsqueeze(0))
    iou = calc_iou(output,label.unsqueeze(0),2)[1]
    print("iou = ",iou)

    out = F.log_softmax(output, dim=1)

    pre_label = out.max(1)[1].squeeze().cpu().data.numpy()

    gt_label = label.squeeze().cpu().data.numpy()

    rgb_img = img.cpu().numpy().transpose(1,2,0)
    label_img = label.cpu().numpy().transpose(1,2,0)


    #
    # imwrite("./models/image_infer/image_infer/img_"+img_path.split("/")[-1],rgb_img[:,:,::-1]*255)
    # imwrite("./models/image_infer/image_infer/mask_"+label_path.split("/")[-1],np.uint8(label_img)*255)
    imwrite("./models/image_infer/infer/unet++/infer_"+label_path.split("/")[-1],np.uint8(pre_label)*255)
    print(img_path)
    print(label_path)

    # map = dense_crf(np.uint8(rgb_img),pre_label,n_labels=2)
    # print(map.shape)
    # print(map == pre_label)

    plt.subplot(1, 3, 1)
    plt.title("ori_img")
    plt.imshow(rgb_img)
    plt.subplot(1, 3, 2)
    plt.title("label_img")
    plt.imshow(gt_label, cmap=plt.cm.gray)
    plt.subplot(1, 3, 3)
    plt.title("predict_img")
    plt.imshow(pre_label, cmap=plt.cm.gray)
    # plt.subplot(1, 4, 4)
    # plt.title("crf_img")
    # plt.imshow(map, cmap=plt.cm.gray)
    plt.pause(0.1)
    plt.show()

    return iou

def convert_mask(img_path,mask_path):

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    cv2.drawContours(image, [cnt], 0, (0, 255, 0), 1)
    cv2.imshow("mask",image)

    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    cv2.imwrite("models/inference/mask_"+img_path.split("/")[-1],image)


def predict_batch(model,img_path,label_path):


    img,label = get_image(img_path,label_path)

    img = img.to(device)
    label = label.to(device)

    output = model(img.unsqueeze(0))
    iou = calc_iou(output,label.unsqueeze(0),2)[1]

    return iou


if __name__ == '__main__':
    # 620 614 608 601 596 594 592 588 570 568 560 541 484 481 398 001
    # 594 592 588 570 484 481 398
    # img_path = "../data10/train/00348.jpg"
    # label_path = "../data10/train_label/00348_label_gt.png"
    # predict_img_path = "models/inference/infer_00004_label_gt.png"

    # [42,61,100]
    # [cju1hmff8tkp809931jps6fbr,cju0sr5ghl0nd08789uzf1raf,cju0roawvklrq0799vmjorwfv]
    img_path = "../kvasir/train/cju1hmff8tkp809931jps6fbr.png"
    label_path = "../kvasir/train_label/cju1hmff8tkp809931jps6fbr.png"
    # 1 4 9 88 125 260 300 330 396 398 544
    # CRFs(img_path,predict_img_path)
    predict(img_path,label_path,in_channels=3,num_classses=2)
    # print("=====================creating Model BANet==================")
    # model = BANet(in_channels=3, num_classes=2)
    # model.eval()
    # model = model.to(device)
    # model_path = "./models/newstodataset_cut_BANet_TRAIN/model_189.pth"
    # model.load_state_dict(torch.load(model_path, map_location='cpu'))
    #
    # path = "C:/Users/admin/Desktop/data02/test"
    # l_path = "C:/Users/admin/Desktop/data02/test_label"
    # path_mod = "C:/Users/admin/Desktop/data02/test_mod"
    # l_path_mod = "C:/Users/admin/Desktop/data02/test_label_mod"
    # img_mod = []
    # for imgname in  os.listdir(path):
    #     img_path = os.path.join(path,imgname)
    #     img_mod_path = os.path.join(path_mod,imgname)
    #     labelname = imgname[:-4]+"_label_gt.png"
    #     label_path = os.path.join(l_path,labelname)
    #     label_mod_path = os.path.join(l_path_mod,labelname)
    #     # print(img_path,label_path)
    #     iou = predict_batch(model,img_path,label_path)
    #     if(iou<0.5):
    #         print(img_path +"====>"+img_mod_path)
    #         print(label_path +"====>"+label_mod_path)
    #         shutil.move(img_path,img_mod_path)
    #         shutil.move(label_path,label_mod_path)
    #         img_mod.append(img_path)
    # print("一共有{}张图片".format(len(img_mod)))









