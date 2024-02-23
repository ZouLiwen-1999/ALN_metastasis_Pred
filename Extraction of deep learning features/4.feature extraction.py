import torch
import torch.nn as nn
import torchvision
import os
join=os.path.join
from torchvision import transforms
from dataset import MyDataset
import csv 

# Function to extract deep learning features
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs

def choose_model(model_name,device,pretrained=True,model_weight=None,class_num=2):
    model_names=['resnet50','resnet101','vgg16','densenet121','effnetb3']
    assert model_name in model_names
    if model_name=='resnet50':
        net = torchvision.models.resnet.resnet50(pretrained=pretrained) 
        inchannel = net.fc.in_features
        net.fc = nn.Linear(inchannel, class_num)
    elif model_name=='resnet101':
        net = torchvision.models.resnet.resnet101(pretrained=pretrained)
        inchannel = net.fc.in_features
        net.fc = nn.Linear(inchannel, class_num)
    elif model_name=='vgg16':
        net = torchvision.models.vgg.vgg16(pretrained=pretrained)
        inchannel = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(inchannel, class_num)
    if model_weight is not None:
        net.load_state_dict(torch.load(model_weight), strict=False)
    net.to(device)
    return net

def itest(model_name,model, model_weight_path,num_class):

    data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    test_dataset = MyDataset(data_root,"test",
                                     transform=data_transform)
    batch_size = 1
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers   
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=nw)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))  # Load the best training weights
    model.eval()

    extract_list = ["conv1", "maxpool", "layer1", "avgpool", "fc"]
    extract_result = FeatureExtractor(model, extract_list)
    feature_dim=2048

# Save deep learning features
    with open(join(data_root,model_name+'feature.csv'),newline='',mode='w') as f:
        csv_writer=csv.writer(f)
        title=['file']
        for i in range(feature_dim):
            title.append('f_'+str(i+1))
        csv_writer.writerow(title)
        for i, (inputs,labels,names) in enumerate(test_loader):
            inputs = inputs.cuda()
            name=names[0]
            outputs = model(inputs)
            feature=extract_result(inputs)[3]
            feature=feature.squeeze(-1)
            feature=feature.squeeze(-1)
            feature=feature.squeeze(0)
            feature = feature.cpu().detach().numpy()
            features=[name]
            for i in range(feature_dim):
                features.append(feature[i])
            csv_writer.writerow(features)
            print(name,feature.shape)

if __name__ == '__main__':
    data_root = '------'   # Path of cut images
    model_name='resnet50'
    class_num=2
    model_weight_path = '------.pth' # Path of best model weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = choose_model(model_name,device,pretrained=False,model_weight=model_weight_path,class_num=2)
    model = model.to(device)
    itest(model_name,model, model_weight_path,num_class=class_num)
