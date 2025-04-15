# Functions here are taken from 
#https://github.com/aradha/recursive_feature_machines
import torch
from torch.autograd import Variable
import torch.optim as optim
import time
#import model1
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os

def visualize_M(M, idx):
    d, _ = M.shape
    SIZE = int(np.sqrt(d // 3))
    F1 = np.diag(M[:SIZE**2, :SIZE**2]).reshape(SIZE, SIZE)
    F2 = np.diag(M[SIZE**2:2*SIZE**2, SIZE**2:2*SIZE**2]).reshape(SIZE, SIZE)
    F3 = np.diag(M[2*SIZE**2:, 2*SIZE**2:]).reshape(SIZE, SIZE)
    F = np.stack([F1, F2, F3])
    print(F.shape)
    F = (F - F.min()) / (F.max() - F.min())
    F = np.rollaxis(F, 0, 3)
    plt.imshow(F)
    plt.axis('off')
    plt.savefig('./video_logs/' + str(idx).zfill(6) + '.png',
                bbox_inches='tight', pad_inches = 0)
    return F


def train_network(train_loader, val_loader, test_loader, net, optimizer, lfn, root_path,
                  num_classes=2, name=None, num_epochs = 5, 
                  save_frames=False):


    #for idx, batch in enumerate(train_loader):
        #inputs, labels = batch
        #_, dim = inputs.shape
        #break
    #net = neural_model.Net(dim, num_classes=num_classes)

    params = 0
    for idx, param in enumerate(list(net.parameters())):
        size = 1
        for idx in range(len(param.size())):
            size *= param.size()[idx]
            params += size
    print("NUMBER OF PARAMS: ", params)

    net.cuda()
    best_val_acc = 0
    best_test_acc = 0
    #best_val_loss = np.float("inf")
    best_val_loss = float("inf")
    best_test_loss = 0
    os.makedirs(root_path, exist_ok=True)     
    for i in range(num_epochs):
        if save_frames:
            net.cpu()
            for idx, p in enumerate(net.parameters()):
                if idx == 0:
                    M = p.data.numpy()
            M = M.T @ M
            visualize_M(M, i)
            net.cuda()

        if i == 0 or i == 1:
            net.cpu()
            d = {}
            d['state_dict'] = net.state_dict()    
            if name is not None:
                file_path = os.path.join(root_path, f'{name}_trained_nn_{i}.pth')
            else:
                file_path = os.path.join(root_path, f'trained_nn_{i}.pth')
            torch.save(d, file_path)
            net.cuda()

        train_loss = train_step(net, optimizer, lfn, train_loader, save_frames=save_frames)
        val_loss = val_step(net, val_loader, lfn)
        test_loss = val_step(net, test_loader, lfn)
        
        #if (isinstance(lfn, nn.CrossEntropyLoss) or isinstance(lfn, nn.NLLLoss)):
        train_acc = get_acc_ce(net, train_loader)
        val_acc = get_acc_ce(net, val_loader)
        test_acc = get_acc_ce(net, test_loader)
        #elif(isinstance(lfn, nn.MSELoss)):
            #train_acc = get_acc_mse(net, train_loader)
            #val_acc = get_acc_mse(net, val_loader)
            #test_acc = get_acc_mse(net, test_loader)
            

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            net.cpu()
            d = {}
            d['state_dict'] = net.state_dict()
            if name is not None:
                file_path = os.path.join(root_path, f'{name}_trained_nn.pth')
            else:
                file_path = os.path.join(root_path, f'trained_nn.pth')
            torch.save(d, file_path)
            net.cuda()

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss

        print("Epoch: ", i,
              "Train Loss: ", train_loss, "Test Loss: ", test_loss,
              "Train Acc: ", train_acc, "Test Acc: ", test_acc,
              "Best Val Acc: ", best_val_acc, "Best Val Loss: ", best_val_loss,
              "Best Test Acc: ", best_test_acc, "Best Test Loss: ", best_test_loss)


def get_data(loader):
    X = []
    y = []
    for idx, batch in enumerate(loader):
        inputs, labels = batch
        X.append(inputs)
        y.append(labels)
    return torch.cat(X, dim=0), torch.cat(y, dim=0)


def train_step(net, optimizer, lfn, train_loader, save_frames=False):
    net.train()
    start = time.time()
    train_loss = 0.

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = batch
        targets = labels
        output = net(Variable(inputs).cuda())
        target = Variable(targets).cuda()
        #loss = torch.mean(torch.pow(output - target, 2))
        loss= lfn(output,target)
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().data.numpy() * len(inputs)
    end = time.time()
    print("Time: ", end - start)
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss


def val_step(net, val_loader, lfn):
    net.eval()
    val_loss = 0.

    for batch_idx, batch in enumerate(val_loader):
        inputs, labels = batch
        targets = labels
        with torch.no_grad():
            output = net(Variable(inputs).cuda())
            target = Variable(targets).cuda()
        #loss = torch.mean(torch.pow(output - target, 2))
        loss= lfn(output,target)
        val_loss += loss.cpu().data.numpy() * len(inputs)
    val_loss = val_loss / len(val_loader.dataset)
    return val_loss

#TODO: Handle a single output net with MSE by rounding it. Unused function
def get_acc_mse(net, loader):
    # This assumes the output of the network is the number of classes and the targets are one-hot vectors
    net.eval()
    count = 0
    for batch_idx, batch in enumerate(loader):
        inputs, targets = batch
        with torch.no_grad():
            #Variable is depreceated, use tensor
            output = net(Variable(inputs).cuda())
            target = Variable(targets).cuda()

        preds = torch.argmax(output, dim=-1)
        labels = torch.argmax(target, dim=-1)

        count += torch.sum(labels == preds).cpu().data.numpy()
    return count / len(loader.dataset) * 100

def get_acc_ce(net, loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()  # Move to CUDA consistently
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)  # Get predicted classes
            total += targets.size(0)
            # Targets maybe in one-hot format. Hence Max
            if len(targets.size()) > 1:
                _, labels = torch.max(targets, -1)
            else:
                labels = targets
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def visualize_predictions(model, dataloader, class_names, device="cpu", num_images=4):
    """
    Visualizes a batch of images from a DataLoader, along with their predicted
    and actual labels, in a grid layout.

    Args:
        model (torch.nn.Module): The trained neural network model.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing the images and labels.
        class_names (list): A list of class names corresponding to the label indices.
        device (str, optional): The device to use for computation ('cpu' or 'cuda'). Defaults to 'cpu'.
        num_images (int, optional): The number of images to visualize from the batch. Defaults to 4.
    """
    model.to(device)  # Ensure model is on the correct device
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation for inference
        try:
            images, labels = next(iter(dataloader))  # Get a batch of data
        except StopIteration:
            print("DataLoader is empty.")
            return

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)  # Get the model's predictions
        _, predicted = torch.max(outputs, 1)  # Get the predicted class indices

    # Convert tensors to numpy arrays for visualization
    images_np = images.cpu().numpy()
    labels_np = labels.cpu().numpy()
    predicted_np = predicted.cpu().numpy()

    # Ensure we don't try to display more images than are in the batch
    num_images = min(num_images, images_np.shape[0])

    # Calculate the number of rows and columns for the grid
    num_cols = int(np.ceil(np.sqrt(num_images)))  # Determine the number of columns
    num_rows = int(np.ceil(num_images / num_cols))  # Determine the number of rows

    plt.figure(figsize=(15, 3 * num_rows))  # Adjust figure size based on the number of rows

    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)  # Create subplots in a grid layout

        # images_np[i] has shape (C, H, W). Transpose to (H, W, C) for imshow.
        img = images_np[i].transpose((1, 2, 0))

        # If the image has been normalized, unnormalize it. This is VERY important.
        # The normalization depends on how you normalized the image in your dataset.
        # Common normalization:
        # transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # If you used this, unnormalize with:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)  # Ensure values are in the range [0, 1]

        plt.imshow(img)

        # Determine the color of the text based on whether the prediction is correct
        if labels_np[i] == predicted_np[i]:
            color = 'green'
        else:
            color = 'red'

        # Align the text for better readability
        plt.title(f"Expected: {class_names[labels_np[i]]}\nPredicted: {class_names[predicted_np[i]]}", 
        color=color, loc='center')
        plt.axis('off')  # Turn off axis labels

    plt.tight_layout()  # Adjust layout to prevent overlapping titles
    plt.show()
