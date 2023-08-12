import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from models import densenet,resnet,baseline,CAM,inception,se_resnet,CAM_resnet
from torchvision import transforms
from tqdm import tqdm
from skimage import io
from matplotlib import cm
import cv2
import random

colormap = cm.get_cmap('inferno')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

CAM_GRID = 0
GRAD_CAM_GRAD = 0
GRAD_CAM_FMAP = 0

def load_model_for_eval(model_path, model_type, labels):

    if(model_type=='resnet') : model = resnet(labels)
    elif(model_type=='densenet') : model = densenet(labels)
    elif(model_type=='inception') : model = inception(labels)
    elif(model_type=='se_resnet') : model = se_resnet(labels)
    elif(model_type=='baseline') : model = baseline(labels)
    elif(model_type=='CAM') : model = CAM(labels)
    elif(model_type=='CAM_resnet') : model = CAM_resnet(labels)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    return model

def load_image(image_path):

    img = transform(cv2.imread(image_path))
    img = img.unsqueeze(0)
    return img    

def occlusion(model_path, model_type, image_path, labels, window_size = 8):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model_for_eval(model_path, model_type, labels)

    img = load_image(image_path)

    results = np.zeros((labels , img.shape[2], img.shape[3]))

    model.to(device)

    for i in tqdm(range(0,img.shape[2]-1,window_size)):
        for j in range(0,img.shape[3]-1,window_size):

            img_ = torch.clone(img)
            img_ = img_.to(device)
            img_[:,:,i:i+window_size,j:j+window_size] = 0

            output = F.softmax(model(img_))
    
            for k in range(labels):

                results[k ,i:i+window_size, j:j+window_size] = output.cpu().data[0][k]

    for k in range(labels):
   
        io.imsave("Results//occulsion_{}.png".format(k), colormap((results[k]*255).astype(np.uint8)))

def hook_forward_fmap(module, input, output):

    output = output.squeeze(0)
    output = output.cpu().data
    output_idx = torch.randint(0, output.shape[0]-1, (1,9))
    output = output[output_idx]
    output = output.squeeze(0)

    plt.figure(figsize=(30,30))

    for i in range(9):
        plt.subplot(330+i+1)
        plt.imshow(output[i])

    plt.savefig('Results\\feature_maps.png')

def hook_forward_CAM_fmap(module, input, output):

    global CAM_GRID
    CAM_GRID = output

def grad_cam_backward(module, grad_input, grad_output):

    global GRAD_CAM_GRAD
    if(GRAD_CAM_FMAP.shape == grad_output[0].squeeze(0).shape) : GRAD_CAM_GRAD = grad_output[0].squeeze(0)

def grad_cam_forward(module, input, output):

    global GRAD_CAM_FMAP
    GRAD_CAM_FMAP = torch.relu(output[0])

def feature_maps_resnet(model_path, image_path, labels, method = 'shallow'):

    index = random.randint(0,8)

    model = load_model_for_eval(model_path, 'resnet', labels)

    if(method=='shallow') :
        
        target_layer = model.pretrained_model.conv1

    else :
        
        layers = [model.pretrained_model.layer1[0].conv3,model.pretrained_model.layer1[1].conv3,model.pretrained_model.layer1[2].conv3,
              model.pretrained_model.layer2[0].conv3,model.pretrained_model.layer2[1].conv3,model.pretrained_model.layer2[2].conv3,
              model.pretrained_model.layer3[0].conv3,model.pretrained_model.layer3[1].conv3,model.pretrained_model.layer3[2].conv3]
    
        target_layer = layers[index]

    img = load_image(image_path)

    hook_layer = target_layer.register_forward_hook(hook_forward_fmap)
    output = model(img)

    hook_layer.remove()

def feature_maps_densenet(model_path, image_path, labels, method = 'deep'):

    index = random.randint(0,8)

    model = load_model_for_eval(model_path, 'densenet', labels)

    if(method=='shallow') :
        
        target_layer = model.pretrained_model.features.conv0

    else :
        
        layers = [model.pretrained_model.features.denseblock1.denselayer1.conv2,model.pretrained_model.features.denseblock1.denselayer2.conv2,model.pretrained_model.features.denseblock1.denselayer3.conv2,
                    model.pretrained_model.features.denseblock2.denselayer1.conv2,model.pretrained_model.features.denseblock1.denselayer2.conv2,model.pretrained_model.features.denseblock1.denselayer3.conv2,
                    model.pretrained_model.features.denseblock3.denselayer1.conv2,model.pretrained_model.features.denseblock1.denselayer2.conv2,model.pretrained_model.features.denseblock1.denselayer3.conv2]
    
        target_layer = layers[index]

    img = load_image(image_path)

    hook_layer = target_layer.register_forward_hook(hook_forward_fmap)
    output = model(img)

    hook_layer.remove()

def viz_cam(model_path, img_path, labels):

    img = load_image(img_path)
    if(labels==10) : 
        model = load_model_for_eval(model_path, 'CAM', labels)
        weights = model.fc1.weight
        hook_layer = model.conv5.register_forward_hook(hook_forward_CAM_fmap)
    elif(labels==2) : 
        model = load_model_for_eval(model_path, 'CAM_resnet', labels)
        weights = model.fc.weight
        hook_layer = model.pretrained_model[-1][-1].relu.register_forward_hook(hook_forward_CAM_fmap)
    
    img = img.to('cuda')
    model = model.to('cuda')

    if(labels == 10) :

        plt.figure(figsize=(40,20))

        for i in range(10):

            plt.subplot(2, 5, i+1)
            output = model(img)
            output = F.softmax(output, dim = 1)[0]
            cam = (weights[i,:] * CAM_GRID.squeeze(0).permute(1,2,0)).sum(2).cpu().data
            plt.imshow(cam)
            plt.title("Probability of {} : {}".format(i , output[i].cpu().data))

    else:

        plt.figure(figsize=(40,20))   

        for i in range(2):

            plt.subplot(1, 2, i+1)
            output = model(img)
            output = F.softmax(output, dim = 1)[0]
            cam = (weights[i,:] * CAM_GRID.squeeze(0).permute(1,2,0)).sum(2).cpu().data
            cam = F.interpolate(input = cam.unsqueeze(0).unsqueeze(0), size = (224,224), mode='bilinear').squeeze(0).squeeze(0)
            plt.imshow(cam)
            plt.title("Probability of {} : {}".format(i , output[i].cpu().data))     

    hook_layer.remove()   
    if(labels==10) : plt.savefig('Results//CAM_mnist.png') 
    else : plt.savefig('Results//CAM_petimages.png') 

class GuidedBackprop:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.target_layer = target_layer
        self.attach_hooks()

    def attach_hooks(self):
        def hook_fn(module, grad_input, grad_output):
            if isinstance(module, torch.nn.ReLU):
                return (torch.clamp(grad_input[0], min=0.0),)

        handle = self.target_layer.register_backward_hook(hook_fn)

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_image, target_class):
        self.model.zero_grad()
        output = self.model(input_image)
        
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class] = 1
        output.backward(gradient=one_hot_output)

        gradients = input_image.grad.clone()
        self.gradients = gradients.squeeze()
        input_image.grad.zero_()

    def guided_backpropagation(self):
        return self.gradients
    
def guided_backprop_resnet(model_path, image_path, target_label, labels):

    img = load_image(image_path)
    img.requires_grad_()

    model = load_model_for_eval(model_path, 'resnet', labels)

    target_layers = [model.pretrained_model.relu] + [model.pretrained_model.layer4[i].relu for i in range(3)] + [model.pretrained_model.layer3[i].relu for i in range(23)] + [model.pretrained_model.layer2[i].relu for i in range(4)] + [model.pretrained_model.layer1[i].relu for i in range(3)] 
    output_idx = np.random.randint(0, len(target_layers)-1, (9,))
    
    plt.figure(figsize=(30,30))
    count = 0

    for i in output_idx: 

        guided_backprop = GuidedBackprop(model, target_layers[i])
        guided_backprop(img, target_label)

        X = guided_backprop.guided_backpropagation().permute(1,2,0).cpu().data
        for d in range(3):
            X[:,:,d] = (X[:,:,d] - X[:,:,d].min()) / (X[:,:,d].max() - X[:,:,d].min())
        plt.subplot(330+count+1)
        count+=1
        plt.imshow(guided_backprop.gradients.permute(1,2,0).cpu().data)

    plt.savefig('Results\\guided_backprop.png')

def grad_cam(model_path, model_type, labels, label, image_path):

    global GRAD_CAM_FMAP
    global GRAD_CAM_GRAD

    img = load_image(image_path)
    model = load_model_for_eval(model_path, model_type, labels)

    target_layer = model.conv3
    target_layer.register_backward_hook(grad_cam_backward)
    target_layer.register_forward_hook(grad_cam_forward)

    output = model(img).squeeze(0)
    output[label].backward()

    GRAD_CAM_GRAD = torch.relu(nn.AdaptiveAvgPool2d(output_size=(1, 1))(GRAD_CAM_GRAD).squeeze(1).squeeze(1))
    GRAD_CAM_sigmoid = torch.sigmoid(GRAD_CAM_GRAD)
    GRAD_CAM_softmax = F.softmax(GRAD_CAM_GRAD, dim = 0)

    plt.figure(figsize=(20,10))

    plt.subplot(121)
    plt.imshow((GRAD_CAM_sigmoid * GRAD_CAM_FMAP.permute(1,2,0)).sum(2).cpu().data)
    plt.subplot(122)
    plt.imshow((GRAD_CAM_softmax * GRAD_CAM_FMAP.permute(1,2,0)).sum(2).cpu().data)

    plt.savefig("Results\\grad_cam.png")

grad_cam("models\\models_mnist\\baseline.pth",
         "baseline",
         10,
         5,
         "sample_images\\mnist_5.jpg")