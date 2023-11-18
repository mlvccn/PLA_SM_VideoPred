import random
import cv2
from PIL import Image
import numpy as np
from matplotlib.path import Path
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import torch.nn as nn
import torch
from MinkowskiEngine import SparseTensor
import torch.nn.functional as F
import math
import torchvision.transforms as transforms



def transforms_video(video):
    # 随机数据增强
    v = random.random()
    # 顺序可以自己随意调整
    T, C, H, W = video.shape
    if v < 0.5:
        for i in range(T):
            video[i] = random_crop_img(video[i],target_size=H)
            video[i] = flip_image(video[i])
    return video
    

def flip_image(img):
    # 将图片随机左右翻转， 根据需要也可以设置随机上下翻转
    v = random.random()
    if v < 0.5:
        img = cv2.flip(img, 1)
    return img

def random_crop(image, min_ratio=0.6, max_ratio=1.0):

    h, w = image.shape[:2]
    
    ratio = random.random()
    
    scale = min_ratio + ratio * (max_ratio - min_ratio)
    
    new_h = int(h*scale)    
    new_w = int(w*scale)
    
    y = np.random.randint(0, h - new_h)    
    x = np.random.randint(0, w - new_w)
    
    image = image[y:y+new_h, x:x+new_w, :]
    image = cv2.resize(image, h, w)
    return image

def random_crop_img(img, target_size, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.shape[1]) / img.shape[0]) / (w**2),
                (float(img.shape[0]) / img.shape[1]) / (h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.shape[1] * img.shape[0] * np.random.uniform(scale_min,
                                                                scale_max)
    target_s = math.sqrt(target_area)
    w = int(target_s * w)
    h = int(target_s * h)
    i = np.random.randint(0, img.shape[1] - w + 1)
    j = np.random.randint(0, img.shape[0] - h + 1)

    img = img[i:i+w, j:j+h]
    img = cv2.resize(img, (int(target_size), int(target_size)))
    return img



class MinkowskiGRN(nn.Module):
    """ GRN layer for sparse tensors.
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key

        Gx = torch.norm(x.F, p=2, dim=0, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return SparseTensor(
                self.gamma * (x.F * Nx) + self.beta + x.F,
                coordinate_map_key=in_key,
                coordinate_manager=cm)

class MinkowskiDropPath(nn.Module):
    """ Drop Path for sparse tensors.
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(MinkowskiDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key
        keep_prob = 1 - self.drop_prob
        mask = torch.cat([
            torch.ones(len(_)) if random.uniform(0, 1) > self.drop_prob
            else torch.zeros(len(_)) for _ in x.decomposed_coordinates
        ]).view(-1, 1).to(x.device)
        if keep_prob > 0.0 and self.scale_by_keep:
            mask.div_(keep_prob)
        return SparseTensor(
                x.F * mask,
                coordinate_map_key=in_key,
                coordinate_manager=cm)

class MinkowskiLayerNorm(nn.Module):
    """ Channel-wise layer normalization for sparse tensors.
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
    ):
        super(MinkowskiLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)
    def forward(self, input):
        output = self.ln(input.F)
        return SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager)

class MinkowskiGroupNorm(nn.Module):
    """ Channel-wise layer normalization for sparse tensors.
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
    ):
        super(MinkowskiGroupNorm, self).__init__()
        self.gn = nn.GroupNorm(2,normalized_shape)#nn.LayerNorm(normalized_shape, eps=eps)
    def forward(self, input):
        output = self.gn(input.F)
        return SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager)


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


###################################Add Noise#################################
def AddPepperNoise(img, s_degree=0.03,t_degree=0.1):  #img[B,T,C,H,W]
    T,C,H,W = img.shape
    if random.uniform(0, 1) < t_degree:
        for i in range(T):
            img_ = np.array(img[i,...]).copy()
            signal_pct = 1 - s_degree
            mask = np.random.choice((0,1,2), size=(1,H,W), p = [signal_pct, s_degree/2.0, s_degree/2.0])
            #mask = np.repeat(mask, C, axis = 1)
            img_[mask == 1] = 1
            img_[mask == 2] = 0
            img[i,...] = img_
    # img_draw = np.ones((H,T*W,C))
    # res_height = H
    # res_width = W
    # for i in range(T):
    #     img_draw[:res_height,i * res_width : (i+1) * res_width,:] = img[i,...].transpose(1,2,0)
    # img_draw = np.maximum(img_draw, 0)
    # img_draw = np.minimum(img_draw, 1)
    # cv2.imwrite('img_noise'+'.png', (img_draw * 255).astype(np.uint8))
    return img

def AddGaussianNoise(img, mean=0.5, sigma=0.5,t_degree=0.01):  #img[B,T,C,H,W]
    T,C,H,W = img.shape
    if random.uniform(0, 1) < t_degree:
        for i in range(T):
            img_ = np.array(img[i,...]).copy()
            noise = np.random.normal(mean,sigma,img_.shape)
            gaussian_out = img_ + noise
            gaussian_out = np.clip(gaussian_out,0,1)
            img[i,...] = gaussian_out
    # img_draw = np.ones((H,T*W,C))
    # res_height = H
    # res_width = W
    # for i in range(T):
    #     img_draw[:res_height,i * res_width : (i+1) * res_width,:] = img[i,...].transpose(1,2,0)
    # img_draw = np.maximum(img_draw, 0)
    # img_draw = np.minimum(img_draw, 1)
    # cv2.imwrite('img_noise'+'.png', (img_draw * 255).astype(np.uint8))
    return img

def AddUniformNoise(img, a=0, b=1):  #img[B,T,C,H,W]
    T,C,H,W = img.shape
    if random.uniform(0, 1) < t_degree:
        for i in range(T):
            img_ = np.array(img[i,...]).copy()
            noise = np.random.uniform(a,b,img_.shape)
            uniform_out = img_ + noise
            uniform_out = np.clip(uniform_out,0,1)
            img[i,...] = uniform_out
    # img_draw = np.ones((H,T*W,C))
    # res_height = H
    # res_width = W
    # for i in range(T):
    #     img_draw[:res_height,i * res_width : (i+1) * res_width,:] = img[i,...].transpose(1,2,0)
    # img_draw = np.maximum(img_draw, 0)
    # img_draw = np.minimum(img_draw, 1)
    # cv2.imwrite('img_noise'+'.png', (img_draw * 255).astype(np.uint8))
    return img
            

        



# ###########################################################################
# Create masks with random shape
# ###########################################################################


def create_random_shape_with_random_motion(video_length,
                                           imageHeight=240,
                                           imageWidth=432):
    # get a random shape
    height = random.randint(imageHeight // 3, imageHeight - 1)
    width = random.randint(imageWidth // 3, imageWidth - 1)
    edge_num = random.randint(6, 8)
    ratio = random.randint(6, 8) / 10
    region = get_random_shape(edge_num=edge_num,
                              ratio=ratio,
                              height=height,
                              width=width)
    region_width, region_height = region.size
    # get random position
    x, y = random.randint(0, imageHeight - region_height), random.randint(
        0, imageWidth - region_width)
    velocity = get_random_velocity(max_speed=3)
    m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
    m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
    #将图像粘贴到另一个图像上,4元组表示粘贴的位置
    masks = [m.convert('L')]
    #转换成灰度图
    # return fixed masks
    if random.uniform(0, 1) > 0.5:
        return masks * video_length
    # return moving masks  这种情况mask不移动
    for _ in range(video_length - 1):
        x, y, velocity = random_move_control_points(x,
                                                    y,
                                                    imageHeight,
                                                    imageWidth,
                                                    velocity,
                                                    region.size,
                                                    maxLineAcceleration=(3,
                                                                         0.5),
                                                    maxInitSpeed=3)
        m = Image.fromarray(
            np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
        masks.append(m.convert('L'))
    return masks


def get_random_shape(edge_num=9, ratio=0.7, width=432, height=240):
    '''
      There is the initial point and 3 points per cubic bezier curve.
      Thus, the curve will only pass though n points, which will be the sharp edges.
      The other 2 modify the shape of the bezier curve.
      edge_num, Number of possibly sharp edges
      points_num, number of points in the Path
      ratio, (0, 1) magnitude of the perturbation from the unit circle,
    '''
    points_num = edge_num * 3 + 1
    angles = np.linspace(0, 2 * np.pi, points_num)
    #在0到2Π构建points_num个等差数列
    codes = np.full(points_num, Path.CURVE4)

    codes[0] = Path.MOVETO
    #生成对应的路径
    # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
    verts = np.stack((np.cos(angles), np.sin(angles))).T * \
        (2*ratio*np.random.random(points_num)+1-ratio)[:, None]
    verts[-1, :] = verts[0, :]   #首尾相接称为一个封闭图形
    path = Path(verts, codes)
    # draw paths into images
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='black', lw=2)
    ax.add_patch(patch)
    ax.set_xlim(np.min(verts) * 1.1, np.max(verts) * 1.1)
    ax.set_ylim(np.min(verts) * 1.1, np.max(verts) * 1.1)
    ax.axis('off')  # removes the axis to leave only the shape
    fig.canvas.draw()
    #plt.savefig('zz.png')
    # convert plt images into numpy images
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #Matplotlib图以numpy数组形式成像
    data = data.reshape((fig.canvas.get_width_height()[::-1] + (3, )))
    plt.close(fig)
    # postprocess
    data = cv2.resize(data, (width, height))[:, :, 0]
    data = (1 - np.array(data > 0).astype(np.uint8)) * 255
    corrdinates = np.where(data > 0)
    xmin, xmax, ymin, ymax = np.min(corrdinates[0]), np.max(
        corrdinates[0]), np.min(corrdinates[1]), np.max(corrdinates[1])
    region = Image.fromarray(data).crop((ymin, xmin, ymax, xmax))
    return region
    #(224,56)


def random_accelerate(velocity, maxAcceleration, dist='uniform'):
    speed, angle = velocity
    d_speed, d_angle = maxAcceleration
    if dist == 'uniform':
        speed += np.random.uniform(-d_speed, d_speed)
        angle += np.random.uniform(-d_angle, d_angle)
    elif dist == 'guassian':
        speed += np.random.normal(0, d_speed / 2)
        angle += np.random.normal(0, d_angle / 2)
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    return (speed, angle)


def get_random_velocity(max_speed=3, dist='uniform'):
    if dist == 'uniform':
        speed = np.random.uniform(max_speed)
    elif dist == 'guassian':
        speed = np.abs(np.random.normal(0, max_speed / 2))
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    angle = np.random.uniform(0, 2 * np.pi)
    return (speed, angle)


def random_move_control_points(X,
                               Y,
                               imageHeight,
                               imageWidth,
                               lineVelocity,
                               region_size,
                               maxLineAcceleration=(3, 0.5),
                               maxInitSpeed=3):
    region_width, region_height = region_size
    speed, angle = lineVelocity
    X += int(speed * np.cos(angle))
    Y += int(speed * np.sin(angle))
    lineVelocity = random_accelerate(lineVelocity,
                                     maxLineAcceleration,
                                     dist='guassian')
    if ((X > imageHeight - region_height) or (X < 0)
            or (Y > imageWidth - region_width) or (Y < 0)):
        lineVelocity = get_random_velocity(maxInitSpeed, dist='guassian')
    new_X = np.clip(X, 0, imageHeight - region_height)
    new_Y = np.clip(Y, 0, imageWidth - region_width)
    return new_X, new_Y, lineVelocity