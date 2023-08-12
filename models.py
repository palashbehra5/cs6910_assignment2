import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from se_resnet import se_resnet101

class residual(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(residual, self).__init__()

        self.pool = nn.MaxPool2d(2, 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)         

    def forward(self, x):

        x_ = F.relu(self.pool(self.conv1(x)))
        x_ = F.relu(self.pool(self.conv2(x_)))
        x_ = self.conv3(x_)

        x = x + x_

        return x

class squeeze_excite(nn.Module):

    def __init__(self):
        super(squeeze_excite, self).__init__()

        self.fc1 = nn.Linear(32, 128) 
        self.fc2 = nn.Linear(128, 32)

    def forward(self, x):

        x_ = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)   
        x_ = F.elu(self.fc1(x_))
        x_ = F.elu(self.fc2(x_))
        x_ = torch.sigmoid(x_)

        return x * x_.unsqueeze(-1).unsqueeze(-1)

class baseline(nn.Module):

    def __init__(self, num_labels):
        super(baseline, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 32, 3)
        self.conv5 = nn.Conv2d(32, 16, 3)

        self.fc1 = nn.Linear(576, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_labels)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class se_cnn(nn.Module):

    def __init__(self):
        super(se_cnn, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)    

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)
        self.conv5 = nn.Conv2d(32, 32, 3)

        self.se = squeeze_excite()

        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):

        x = self.pool(F.elu(self.conv1(x)))
        x = self.se(x)

        f1 = self.pool(F.elu(self.conv2(x)))
        x = self.se(f1)

        f2 = self.pool(F.elu(self.conv3(x)))
        x = self.se(f2)

        f3 = self.pool(F.elu(self.conv4(x)))
        x = self.se(f3)

        f4 = self.pool(F.elu(self.conv5(x)))
        x = self.se(f4)

        x = torch.flatten(x, 1)
        # print(x.shape)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class residual_cnn(nn.Module):
    
    def __init__(self):
        super(residual_cnn, self).__init__()

        self.res1 = residual(8, 8)
        self.res2 = residual(8, 8)
        self.res3 = residual(8, 8)
        self.res4 = residual(8, 8)
        self.res5 = residual(8, 8)
        self.res6 = residual(8, 8)
        self.res7 = residual(8, 8)
        self.res8 = residual(8, 8)
        self.res9 = residual(8, 8)
        self.res10 = residual(8, 8)
        self.res11 = residual(8, 8)
        self.res12 = residual(8, 8)
        self.res13 = residual(8, 8)
        self.res14 = residual(8, 8)
        self.res15 = residual(8, 8)
        self.res16 = residual(8, 8)
        self.res17 = residual(8, 8)
        self.res18 = residual(8, 8)
        self.res19 = residual(8, 8)
        self.res20 = residual(8, 8)

        self.pool = nn.MaxPool2d(2, 2)    
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 64, 3)
        self.conv3 = nn.Conv2d(64, 1, 3)

        self.fc1 = nn.Linear(900, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.res9(x)
        x = self.res10(x)
        x = self.res11(x)
        x = self.res12(x)
        x = self.res13(x)
        x = self.res14(x)
        x = self.res15(x)
        x = self.res16(x)
        x = self.res17(x)
        x = self.res18(x)
        x = self.res19(x)
        x = self.res20(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class resnet(nn.Module):
    
    def __init__(self , num_labels):
        super(resnet, self).__init__()

        self.pretrained_model = models.resnet101(pretrained=True)
        self.fc = nn.Linear(1000, num_labels)

    def forward(self, x):

        x = F.relu(self.pretrained_model(x))
        x = self.fc(x)

        return x
    
class vit(nn.Module):

    def __init__(self, num_labels):
        super(vit, self).__init__()

        self.pretrained_model = models.vit_b_16(weights = models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.fc = nn.Linear(1000, num_labels)

    def forward(self, x):

        x = F.relu(self.pretrained_model(x))
        x = self.fc(x)

        return x
    
class densenet(nn.Module):

    def __init__(self, num_labels):
        super(densenet, self).__init__()

        self.pretrained_model = models.densenet201(weights = models.DenseNet201_Weights.IMAGENET1K_V1)
        self.fc = nn.Linear(1000, num_labels)

    def forward(self, x):

        x = F.relu(self.pretrained_model(x))
        x = self.fc(x)

        return x
    
class inception(nn.Module):

    def __init__(self, num_labels):
        super(inception, self).__init__()

        self.pretrained_model =  models.inception_v3(weights = models.Inception_V3_Weights.IMAGENET1K_V1)
        self.fc = nn.Linear(1000, num_labels)

    def forward(self, x):
        
        if(self.training) : x, _ = self.pretrained_model(x)
        else :  x = self.pretrained_model(x)
        x = F.relu(x)
        x = self.fc(x)
        return x
    
class convnext(nn.Module):

    def __init__(self, num_labels):
        super(convnext, self).__init__()

        self.pretrained_model =  models.convnext_large(weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
        self.fc = nn.Linear(1000, num_labels)

    def forward(self, x):
        
        x = self.pretrained_model(x)
        x = F.relu(x)
        x = self.fc(x)
        return x
    
class se_resnet(nn.Module):
    
    def __init__(self, num_labels):
        super(se_resnet, self).__init__()

        self.pretrained_model =  se_resnet101()

        saved_model_path = 'models\\resnet.pth'
        checkpoint = torch.load(saved_model_path)
        res = resnet()
        res.load_state_dict(checkpoint)

        self.pretrained_model.conv1 = res.pretrained_model.conv1
        self.pretrained_model.bn1 = res.pretrained_model.bn1
        self.pretrained_model.relu = res.pretrained_model.relu
        self.pretrained_model.maxpool = res.pretrained_model.maxpool
        self.pretrained_model.layer1 = res.pretrained_model.layer1
        self.pretrained_model.layer2 = res.pretrained_model.layer2
        self.pretrained_model.layer3 = res.pretrained_model.layer3
        self.pretrained_model.layer4 = res.pretrained_model.layer4
        self.pretrained_model.avgpool = res.pretrained_model.avgpool
        self.pretrained_model.fc = res.pretrained_model.fc
        self.fc = res.fc

    def forward(self, x):

        x = F.relu(self.pretrained_model(x))
        x = self.fc(x)

        return x
    
class CAM(nn.Module):

    def __init__(self, num_labels):
        super(CAM, self).__init__()

        self.pool = nn.MaxPool2d(2,1)

        self.conv1 = nn.Conv2d(3,8,3)
        self.conv2 = nn.Conv2d(8,16,3) 
        self.conv3 = nn.Conv2d(16,32,3) 
        self.conv4 = nn.Conv2d(32,64,3) 
        self.conv5 = nn.Conv2d(64,128,3)  

        self.fc1 = nn.Linear(128, num_labels)      

    def forward(self, x):
        
        x = self.pool(F.gelu(self.conv1(x)))
        x = self.pool(F.gelu(self.conv2(x)))
        x = self.pool(F.gelu(self.conv3(x)))
        x = self.pool(F.gelu(self.conv4(x)))
        x = self.pool(F.gelu(self.conv5(x)))

        # Global Average Pooling
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)  

        x = self.fc1(x)

        return x
    
class CAM_resnet(nn.Module):
    
    def __init__(self , num_labels):
        super(CAM_resnet, self).__init__()

        self.pretrained_model =  models.resnet101(weights = models.ResNet101_Weights.IMAGENET1K_V1)
        self.pretrained_model = nn.Sequential(*list(self.pretrained_model.children())[:-3])
        self.fc = nn.Linear(1024, num_labels)

    def forward(self, x):

        x = F.relu(self.pretrained_model(x))

        # Global Average Pooling
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)  

        x = self.fc(x)

        return x

class student(nn.Module):

    def __init__(self, num_labels):

        super(student, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 3)

        self.fc1 = nn.Linear(256 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_labels)

    def forward(self, x):

        x = self.pool(F.gelu(self.conv1(x)))
        x = self.pool(F.gelu(self.conv2(x)))
        x = self.pool(F.gelu(self.conv3(x)))
        x = self.pool(F.gelu(self.conv4(x)))
        x = self.pool(F.gelu(self.conv5(x)))

        x = torch.flatten(x, 1)

        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)

        return x

# X = torch.randint(0, 255, (8, 3, 224, 224)).float()
# net = student(2)
# net.eval()
# print(net(X).shape)