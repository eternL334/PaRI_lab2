import torch
from torchvision.models import vgg13, VGG13_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2 as cv


class VGG13Encoder(torch.nn.Module):
    def __init__(self, num_blocks, weights=VGG13_Weights.DEFAULT):
        super().__init__()
        self.num_blocks = num_blocks
        
        feature_extractor = vgg13(weights=weights).features
        
        self.blocks = torch.nn.ModuleList()
        for idx in range(self.num_blocks):
            self.blocks.append(
               feature_extractor[5 * idx:5 * idx + 4] 
            )

    def forward(self, x):
        activations = []
        for idx, block in enumerate(self.blocks):
            x = block(x)

            activations.append(x)

            x = torch.functional.F.max_pool2d(x, kernel_size=2, stride=2)
            
        return activations

class DecoderBlock(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.upconv = torch.nn.Conv2d(
            in_channels=out_channels * 2, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )
        self.conv1 = torch.nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )
        self.relu = torch.nn.ReLU()
        
    def forward(self, down, left):
        x = self.upconv(torch.nn.functional.interpolate(down, scale_factor=2))
        
        x = x + left
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x

class Decoder(torch.nn.Module):
    def __init__(self, num_filters, num_blocks):
        super().__init__()

        self.blocks = torch.nn.ModuleList()
        for idx in range(num_blocks):
            self.blocks.insert(0, DecoderBlock(num_filters * 2 ** idx))   

    def forward(self, acts):
        up = acts[-1]
        for block, left in zip(self.blocks, acts[-2::-1]):
            up = block(up, left)
        return up

class LinkNet(torch.nn.Module):
    def __init__(self, num_classes=1, num_blocks=4):
        super().__init__()
        self.encoder = VGG13Encoder(num_blocks)
        
        self.decoder = Decoder(64, num_blocks - 1)
        
        self.final = torch.nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final(x)

        return x

class Model:
    def __init__(self, weights_path):
        self.model = LinkNet(num_classes=1, num_blocks=3)
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        self.transform = A.Compose([
            A.Normalize(),
            ToTensorV2(transpose_mask=True)
        ])
    
    def segmentate(self, img: np.ndarray):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = self.transform(image=img)['image'].unsqueeze(0)
        logits = self.model(img)

        mask = (logits.detach().clone().numpy() > 0.0).astype(np.uint8)[0][0]
        kernel = np.ones((13, 13), dtype=np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.erode(mask, kernel)
        return mask
    
    def dots(self, img: np.ndarray):
        sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0)
        sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1)
        sobel_x = cv.convertScaleAbs(sobel_x)
        sobel_y = cv.convertScaleAbs(sobel_y)
        img = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
        img = img.max(axis=2)
        img = cv.GaussianBlur(img,(3,3),0)
        _, img = cv.threshold(img, 50, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return img
    
    def find_triangles(self, img: np.ndarray):
        contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        approxes = []
        for i in range(len(contours)):
            approx = cv.approxPolyDP(contours[i], 0.05 * cv.arcLength(contours[i], True), True)
            if len(approx) == 3:
                approxes.append(approx)
        
        return approxes
    
    def count_dots(self, triangle: np.ndarray, dots_mask: np.ndarray):
        centroid = triangle.mean(axis=0)
        mask_triangle = np.zeros_like(dots_mask)
        cv.drawContours(mask_triangle, [triangle], 0, 1, cv.FILLED)

        counts = []
        for i in range(3): 
            radius = int(np.sum((triangle[i] - centroid) ** 2) ** 0.5)
            mask_circle = np.zeros_like(dots_mask)
            cv.circle(mask_circle, triangle[i], radius, 1, cv.FILLED)

            mask = mask_triangle * mask_circle
            dots_masked = np.where(mask, dots_mask, 0)

            n, labels, stats, centroids = cv.connectedComponentsWithStats(dots_masked)
            counts.append(min(np.sum(stats[1:, -1] > 10), 5))
        
        return centroid.astype(int), counts
    
    def __call__(self, segmentate_mask: np.ndarray, dots_mask: np.ndarray):
        triangles = self.find_triangles(segmentate_mask)

        result = []
        for triangle in triangles:
            centroid, counts = self.count_dots(triangle.squeeze(1), dots_mask)
            result.append((centroid, counts))
        
        return result
