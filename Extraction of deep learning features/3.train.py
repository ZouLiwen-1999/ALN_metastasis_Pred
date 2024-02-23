import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, datasets
from dataset import MyDataset
import json
import os
import torch.optim as optim
from tqdm import tqdm 
import csv

# Choose the basic model
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

def main():

    root_path= '------'  # Path of cut images
    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # Number of workers
    epoch_num=500
    model_name='resnet50'
    save_path = '------.pth'  # Path to save best model weights
    exp_name='exp1'  # Number of experiment

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomRotation([-180, 180]),
                                     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # Load data
    train_dataset = MyDataset(root_path,"train",exp_name,
                                        transform=data_transform["train"])
    train_num = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=nw)

    validate_dataset = MyDataset(root_path, "test", exp_name,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=nw)

    obj_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in obj_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose basic model
    net=choose_model(model_name,device,pretrained=True,model_weight=None,class_num=2)


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # Training
    best_acc = 0.0
    with open(model_name+'_train.csv',newline='',mode='w') as f:
        csv_writer=csv.writer(f)  # Record training results
        csv_writer.writerow(['epoch','loss','acc','best_acc'])
        for epoch in range(epoch_num):
            net.train()
            running_loss = 0.0
            for step, data in enumerate(tqdm(train_loader), start=0):
                images, labels,_ = data
                optimizer.zero_grad()
                logits = net(images.to(device))
                loss = loss_function(logits, labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            net.eval()
            acc = 0.0
            with torch.no_grad():
                for val_data in validate_loader:
                    val_images, val_labels,_ = val_data
                    outputs = net(val_images.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += (predict_y == val_labels.to(device)).sum().item()
                val_accurate = acc / val_num
                if val_accurate > best_acc:
                    best_acc = val_accurate
                    torch.save(net.state_dict(), save_path)  # Save the best model weights
                print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f  best_accuracy: %.3f' %
                    (epoch + 1, running_loss / step, val_accurate, best_acc))
                print()
            csv_writer.writerow([epoch, running_loss / step, val_accurate, best_acc])
    print('Finished Training')

if __name__ == "__main__":
    main()
