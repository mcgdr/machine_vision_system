############################################################################################################################################
!Readme!

This code was written for the "Development of a low-cost machine vision system to recognize and flip the persimmons using deep learning".
The machine vision system is used to recognize and flip the persimmon with a real-time recognition algorithm,
which can be employed in the real persimmon process industry.
Also, this code modifies YOLOv3-tiny using the HSV color space to run a deep learning algorithm in real-time on the NVIDIA Jetson Nano.
It is performed in Python 3.6.5 using PyTorch 1.6.0 and CUDA 10.2.
If you have any help or questions, please send an e-mail to mcgdr@ynu.ac.kr.
############################################################################################################################################


import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
from torch.autograd import Variable
import serial


def make_class(pred_cls):
    if pred_cls == 0:
        return "   lower"
    elif pred_cls == 1:
        return "   stalk"

def make_color_class(pred_cls):
    if pred_cls == 0:
        return (0, 255, 0)
    elif pred_cls == 1:
        return (255, 0, 0)

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def changeBGR2RGB(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # RGB > BGR
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b

    return img

def changeRGB2BGR(img):
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()

    # RGB > BGR
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r

    return img

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())

def MaxPooling(kernel_size=2, stride=2, padding=0):
    return nn.Sequential(
        nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

def UpSampling():
    return nn.Sequential(
        nn.Upsample(scale_factor=2))


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out

class YOLO_tiny(nn.Module):
    def __init__(self, block, num_classes):
        super(YOLO_tiny, self).__init__()

        self.num_classes = num_classes

        self.conv1 = conv_batch(3, 16)
        self.pool = MaxPooling()
        self.conv2 = conv_batch(16, 32)
        self.conv3 = conv_batch(32, 64)
        self.conv4 = conv_batch(64, 128)
        self.conv5 = conv_batch(128, 256)
        self.conv6 = conv_batch(256, 512)
        self.conv7 = conv_batch(512, 1024)
        self.conv8 = conv_batch(1024, 256, kernel_size=1, padding=0)
        # [YOLO]
        self.conv9 = conv_batch(256, 128, kernel_size=1, padding=0)
        self.upsam = UpSampling()
        self.conv10 = conv_batch(384, 256)
        self.conv11 = conv_batch(256, 256, kernel_size=1, padding=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, self.num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.pool(out)
        out = self.conv3(out)
        out = self.pool(out)
        out = self.conv4(out)
        out = self.pool(out)
        out1 = self.conv5(out)
        out = self.pool(out1)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out2 = self.upsam(out)
        out = torch.cat([out1, out2], dim=1)
        out = self.conv10(out)
        out = self.conv11(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 256)
        out = self.fc(out)

        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


def yolotiny(num_classes):
    return YOLO_tiny(DarkResidualBlock, num_classes)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    pred_result = -1
    PORT = '/dev/ttyTHS1' # port to send a signal to the actuator
    BaudRate = 9600
    ARD = serial.Serial(PORT, BaudRate)
    time.sleep(3)

    PATH = 'weight_210_200_99.pth' # path to load weight file
    cap = cv2.VideoCapture(0)
    lower_object = np.array([5, 100, 100])
    upper_object = np.array([20, 255, 255])
    classes = ['lower', 'stalk']
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    net = yolotiny(2)
    net.to(device)
    net.load_state_dict(torch.load(PATH))

    # activating GPU
    res, GPU_ON = cap.read()
    GPU_ON = cv2.resize(GPU_ON, dsize=(416,416), interpolation=cv2.INTER_CUBIC)
    GPU_ON = changeBGR2RGB(GPU_ON)
    GPU_ON = transforms.ToTensor()(GPU_ON)
    GPU_ON, _ = pad_to_square(GPU_ON, 0)
    GPU_ON = resize(GPU_ON, 416)
    GPU_ON = GPU_ON.unsqueeze(0)
    GPU_ON = Variable(GPU_ON.type(Tensor))

    with torch.no_grad():
        outputs = net(GPU_ON)
        _, predicted = torch.max(outputs.data, 1)
    predicted = 0

    resol_size = 800

    # calculating average FPS
    FPS_count = 0
    FPSs = 0
    infers = 0
    
    while(1):

        res, frame = cap.read()

        start = time.time()


        # convert BGR to HSV
        img_original = cv2.resize(frame, dsize=(resol_size, resol_size), interpolation=cv2.INTER_CUBIC)
        frame = cv2.resize(frame, dsize=(resol_size, resol_size), interpolation=cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_object = cv2.inRange(hsv, lower_object, upper_object)
        masking = cv2.bitwise_and(frame, frame, mask=mask_object)
        binary = cv2.cvtColor(masking, cv2.COLOR_BGR2GRAY)
        contour, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        fin_HSV = time.time()

        print("\n+ HSV_convert: %s" % (fin_HSV - start))


        for c in contour:
            # localization part
            if cv2.arcLength(c, True) > 400:
                start_coord_time = time.time()
                x, y, w, h = cv2.boundingRect(c)

                coord = [x - 10, x + w + 10, y - 10, y + h + 10]
                for i in range(0, 4):
                    if coord[i] < 0:
                        coord[i] = 0
                    elif coord[i] > resol_size:
                        coord[i] = resol_size

                x1 = coord[0]
                x2 = coord[1]
                y1 = coord[2]
                y2 = coord[3]

                fin_coord_time = time.time()

                print("\n+ coord : %s" % (fin_coord_time - start_coord_time))


                # classification part
                start_infer_time = time.time()

                cropped_img = cv2.resize(frame[y1: y2, x1: x2], (224, 224), interpolation=cv2.INTER_CUBIC)
                img = changeBGR2RGB(cropped_img)
                input_imgs = transforms.ToTensor()(img)
                input_imgs, _ = pad_to_square(input_imgs, 0)
                input_imgs = input_imgs.unsqueeze(0)
                input_imgs = Variable(input_imgs.type(Tensor))

                with torch.no_grad():
                    outputs = net(input_imgs)
                    _, predicted = torch.max(outputs.data, 1)

                pred_result = predicted[0]
                fin_infer_time = time.time()
                print("\n+ infer : %s" % (fin_infer_time - start_infer_time))


                # representing contour and class
                start_contour_time = time.time()
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=make_color_class(pred_result), thickness=3)
                cv2.putText(frame, make_class(pred_result), org=(x2, y2), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=make_color_class(pred_result), thickness=1)
                fin_contour_time = time.time()
                print("\n+ contour : %s" % (fin_contour_time - start_contour_time))

        # send the signal to the ball screw actuator
        start_act_time = time.time()
        if pred_result == 0:
            state = "3"
            state = state.encode('utf-8')
            ARD.write(state)
        elif pred_result == 1:
            state = "4"
            state = state.encode('utf-8')
            ARD.write(state)

        current_time = time.time()
        print("\n+ activation : %s" % (current_time - start_act_time))


        FPS_count += 1

        if FPS_count >= 1:
            inference_time = current_time - start
            if inference_time == 0:
                inference_time = 0.0000000001
            infers += inference_time
            avg_infer_time = infers / FPS_count
            FPS = 1 / inference_time
            FPSs += FPS
            avg_FPS = FPSs / FPS_count

            print("\n+ %s" % inference_time)
            print("\n+ avg_FPS: %s"% avg_FPS)


        cv2.putText(frame, "FPS : %.5s" % FPS, org=(600, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=1)
        frame_1 = np.hstack((img_original, hsv))
        frame_2 = np.hstack((masking, frame))
        cv2.imshow('result', frame_2)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


    cv2.destroyAllWindows()
