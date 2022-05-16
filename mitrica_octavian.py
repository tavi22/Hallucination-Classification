# Deep Hallucination Classifier - source code
# Mitrica Octavian (241)
# Ran in Kaggle Notebooks

# Libraries
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# PyTorch
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision. models as models

# Image Augmentation
from albumentations.pytorch import transforms
import albumentations

# Metrics
from sklearn.metrics import confusion_matrix


# Set the device
# cpu by default, cuda when GPU is used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Data
def getDataframe(path, test=False):
    '''
    path: path to .txt file
    test: set to True if .txt file is a test set, False otherwise
    return: transformed .txt file to dataframe
    '''

    # Load .txt file
    f = np.loadtxt(path, dtype="str")

    ids_list = []
    labels_list = []

    # Read ID and label (if applicable)
    for row in f:
        row = row.split(",")
        id_col = row[0]
        ids_list.append(id_col)

        if test == False:
            label_col = row[1]
            labels_list.append(label_col)

    # Convert to pandas dataframe and create path variable for convenience
    if test:
        df = pd.DataFrame(data={"id": ids_list[1:]})
        df["path"] = "../input/unibuc-2022-s24/test/" + df["id"]
    else:
        df = pd.DataFrame(data={"id": ids_list[1:], "labels": labels_list[1:]})
        df["path"] = "../input/unibuc-2022-s24/train+validation/" + df["id"]

    return df


# Retrieve data
train = getDataframe('../input/unibuc-2022-s24/train.txt')
valid = getDataframe('../input/unibuc-2022-s24/validation.txt')
test = getDataframe('../input/unibuc-2022-s24/test.txt', test=True)


# Glimpse of the images

def showImages(df, sample_size):
    '''
    df: pandas dataframe
    sample_size: how many images to be displayed (even numbers only as we display them on 2 rows)
    '''

    # Retrieve paths
    paths = df.sample(n=sample_size, random_state=21)["path"].values.tolist()

    # Plot
    fig, axs = plt.subplots(2, int(sample_size/2), figsize=(23, 4))
    axs = axs.flatten()

    for k, path in enumerate(paths):
        img = plt.imread(path)
        axs[k].imshow(img)
        axs[k].axis("off")

    plt.tight_layout()
    plt.show()


showImages(train, sample_size=10)
showImages(valid, sample_size=10)

# Label distribution

# Train Label Distribution
sns.histplot(train["labels"])
# Valid Label Distribution
sns.histplot(valid["labels"])


# Dataset

class ImagesDataset(Dataset):

    def __init__(self, df, train_flag):
        '''Class constructor
        df: the train/valid/test dataframes
        train_flag: True if train or valid data, False otherwise
        :return: returns image and label (for test/valid only) as tensors
        '''

        self.df = df
        self.train_flag = train_flag
        # Albumentations
        if self.train_flag:
            self.transform = albumentations.Compose([
                albumentations.HorizontalFlip(p=0.5),
                albumentations.VerticalFlip(p=0.5),
                albumentations.Rotate(p=0.5),
                albumentations.Normalize()
            ])
        else:
            # Albumentations are applied only during training/validation
            self.transform = albumentations.Compose([
                albumentations.Normalize()
            ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):

        # Retrieve data from dataframe
        rows = self.df.iloc[index]

        # Read image
        image = cv2.imread(rows["path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply albumentations
        transformed_img = self.transform(image=image)['image'].astype(np.float32)
        # Transpose image
        image = transformed_img.transpose(2, 0, 1)
        # Transform image in tensor
        image = torch.tensor(image)

        if self.train_flag:
            # Return the image + it's label only for train/validation (as tensor)
            label = torch.tensor(list(map(int, rows["labels"])))
            return image, label

        else:
            # Return only the image for test data
            return image


# Sanity check - take a look at what we have so far
# Create Dataset object
example_data = ImagesDataset(df=train.head(10), train_flag=True)
# Dataloader - use batches in order to compute loss once every n images
example_loader = DataLoader(example_data, batch_size=5)

for k, (image, label) in enumerate(example_loader):
    print("Batch: ", k)
    print("Image:", image.shape)
    print("Label:", label)


# Create datasets for train/valid/test
train_dataset = ImagesDataset(df=train, train_flag=True)
valid_dataset = ImagesDataset(df=valid, train_flag=True)
test_dataset = ImagesDataset(df=test, train_flag=False)


# MODEL

# Auxiliary functions:
#   computeAccuracy

def computeAccuracy(model, data, batch_size=20):
    '''
    model: our training model
    data: the train/valid data we want to check accuracy for (must be an ImagesDataset instance)
    batch_size: batch size (in order to compute loss once every n images)
    :return: returns accuracy of modelled data as integer
    '''
    # Set the model in eval mode
    model.eval()

    # Create the DataLoader
    # shuffle=True to shuffle images (for train/valid data) after each epoch
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    correct = 0
    total = 0

    for (images, labels) in iter(loader):
        # .to(device) - for GPU support
        images, labels = images.to(device), labels.to(device)

        # Put images through model
        # outputs will hold probability values for each label
        outputs = model(images)

        # Flatten the labels to look just like the predictions
        labels = torch.flatten(labels.cpu().detach())

        # Get max probability and then select only the labels
        # .cpu().detach() - for GPU support again
        # prediction will hold a tensor of the predicted labels
        prediction = outputs.max(dim=1)[1].cpu().detach()

        # Sum when prediction is equal with label and turn into integer
        correct += (prediction == labels).sum().item()

        # Add image to total
        total += images.shape[0]

    # Return accuracy
    return correct / total


#   computePredictions

def computePredictions(model, data):
    '''
    model: our training model
    data: the test data we want the predictions for (must be an ImagesDataset instance)
    :return: returns prediction as tensor
    '''
    # Set the model in eval mode
    model.eval()

    # Create the DataLoader
    # We don't shuffle the test images and batch size is the whole dataset
    loader = DataLoader(data, batch_size=len(data))

    for images in iter(loader):
        # Same as before
        images = images.to(device)
        out = model(images)
        prediction = out.max(dim=1)[1].cpu().detach()

    # returns prediction tensor
    return prediction


#   testModel

def testModel(model, train_data, valid_data, criterion, optimizer, batch_size=20, num_epochs=1):
    '''
    model: our model
    train_data: training data (as ImagesDataset)
    valid_data: validation data (as ImagesDataset)
    criterion: loss function (computes loss)
    optimizer: optimizer function (reduce loss)
    batch_size: batch size
    num_epochs: number of epochs (times we go through the entire data given)
    '''

    print('Loading data...')
    # Get DataLoader for train data in order to train in batches
    # shuffle=True to shuffle images (for train/valid data) after each epoch
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # list of losses - so we can compute loss after each epoch
    losses = []
    # list of average accuracies of each epoch
    trainingAvg = []
    testingAvg = []

    print('Testing the model...')

    # We are testing in epochs (number of times we went over the whole data)
    # to increase average accuracy
    for i in range(num_epochs):
        print(f"---Epoch: {i}---")
        for images, labels in iter(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Set model in training mode:
            model.train()

            # Flatten the labels to look just like the predictions
            labels = torch.flatten(labels.cpu().detach())

            # Put images through model
            outputs = model(images).to(device)

            # Standard procedure
            # Clear gradients from last iteration
            optimizer.zero_grad()

            # Compute loss
            loss = criterion(outputs, labels.to(device))

            # Compute gradients
            loss.backward()

            # Update weights
            optimizer.step()

            losses.append(loss.cpu().detach().numpy().tolist())

            # Compute accuracy after this epoch
            trainingAvg.append(computeAccuracy(model, train_data, batch_size=batch_size))
            testingAvg.append(computeAccuracy(model, valid_data, batch_size=batch_size))


        # Show the last accuracy registered and epoch loss
        print("Epoch Loss:", np.mean(losses))
        print("Training Average Accuracy: ", trainingAvg[-1])
        print("Testing Average Accuracy: ", testingAvg[-1])


# train network

def trainModel(model, train_data, test_data, criterion, optimizer, batch_size=20, num_epochs=1):
    '''
    model: our model
    train_data: training data (as ImagesDataset)
    train_data: train data (as ImagesDataset)
    criterion: loss function (computes loss)
    optimizer: optimizer function (reduce loss)
    batch_size: batch size
    num_epochs: number of epochs (times we go through the entire data given)
    :return: returns the last epochs label prediction
    '''

    print('Loading data...')

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    losses = []
    trainingAvg = []

    print('Training the model...')
    for i in range(num_epochs):
        print(f"---Epoch: {i}---")
        for images, labels in iter(train_loader):
            # same as before
            images, labels = images.to(device), labels.to(device)
            model.train()
            labels = torch.flatten(labels.cpu().detach())
            out = model(images).to(device)
            optimizer.zero_grad()
            loss = criterion(out, labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy().tolist())

            # We append only the train average because we don't actually have
            # labels for the real test data
            trainingAvg.append(computeAccuracy(model, train_data, batch_size=batch_size))



        # Save model
        print("Saving...")
        # We are saving the state_dict which contains all of our model's parameters (weights and biases)
        torch.save(model.state_dict(), f"Final_Epoch_{i}_part2.pt")

        # Show the last accuracy registered and epoch loss
        print("Epoch Loss:", np.mean(losses))
        print("Training Average Accuracy: ", trainingAvg[-1])

    return computePredictions(model, test_data)

# Models and runs
# Simple FNN try - Not submitted (acc too low)

# Creating the FNN
class DeepHallucination_Classifier(nn.Module):

    def __init__(self):
        super(DeepHallucination_Classifier, self).__init__()
        self.layers = nn.Sequential(nn.Linear(3 * 16 * 16, 50),
                                    nn.ReLU(),
                                    nn.Linear(50, 20),
                                    nn.ReLU(),
                                    nn.Linear(20, 7))

    def forward(self, image):
        # Flatten image: from [3, 16, 16] to [764]
        image = torch.reshape(image, (-1, 3 * 16 * 16))
        # Take image through the layers
        outputs = self.layers(image)

        return outputs

# run
# Create a model instance
model = DeepHallucination_Classifier()
model.to(device)

# Create an example dataset with 500 of the train images
example_data = ImagesDataset(df=train.sample(500), train_flag=True)
test_example = ImagesDataset(df=valid.sample(200), train_flag=True)


# Set loss, optimizer and some parameters
learning_rate = 0.001  # how much we update our weights and biases based on the error
weight_decay = 0.0005   # dial back the large weights
num_epochs = 200  # number of epochs
batch_size = 50  # batch size
criterion = nn.CrossEntropyLoss()  # Cross Entropy Function
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)  # Gradient Descent Function

# Test the network
testModel(model, example_data, test_example, criterion, optimizer, batch_size=batch_size, num_epochs=num_epochs)
# Accuracy after 5 epochs: train - 0.234 / valid - 0.26
# Accuracy is low so we don't train the model


# CNN try - FIRST SUBMISSION

# Creating the CNN model
class CNN_DeepHallucination_Classifier(nn.Module):
    def __init__(self):
        super(CNN_DeepHallucination_Classifier, self).__init__()

        # Convolutional Layers
        # Output formula: [(W - K + 2P)/S + 1] x [(W - K + 2P)/S + 1]

        self.features = nn.Sequential(nn.Conv2d(3, 16, 3),  # output: (16-3+0)/1 + 1 = 14
                                      nn.ReLU(),  # activation function
                                      nn.MaxPool2d(2, 2),  # 14/2 = 7
                                      nn.Conv2d(16, 10, 3),  # output: (7-3+0)/1 + 1 = 5
                                      nn.ReLU(),
                                      nn.MaxPool2d(2))  # 5/2 = 2

        # FNN for classification
        self.classifier = nn.Sequential(nn.Linear(10 * 2 * 2, 128),  # 10 channels with 2 * 2 px output
                                        nn.ReLU(),
                                        nn.Linear(128, 84),
                                        nn.ReLU(),
                                        nn.Linear(84, 7))  # 7 possible predictions (labels)

    def forward(self, image):  # image is put through the network

        # convolutions applied to image
        image = self.features(image)

        # flatten image
        image = image.view(-1, 10 * 2 * 2)

        # probabilities output
        outputs = self.classifier(image)

        # applying softmax
        outputs = F.log_softmax(outputs, dim=1)

        return outputs


# run
model = CNN_DeepHallucination_Classifier()
model.to(device)

# Here I loaded up a checkpoint because I did the training
# in 3 parts of 5, 10, 10 epochs
checkpoint = torch.load("../input/cnn-saves/Final_Epoch_9_part2.pt")
model.load_state_dict(checkpoint)

# Some example data to test the network
example_data = ImagesDataset(df=train.sample(500), train_flag=True)
test_example = ImagesDataset(df=valid.sample(200), train_flag=True)


# All of the data
train_data = ImagesDataset(df=train, train_flag=True)
test_data = ImagesDataset(df=test, train_flag=False)

# Submission parameters
learning_rate = 0.001  # how much we update our weights and biases based on the error
weight_decay = 0.0005   # dial back the large weights
batch_size = 200  # batch size
num_epochs = 25  # number of epochs
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Adam optimizer function

testModel(model, example_data, test_example, criterion=criterion,
          optimizer=optimizer, batch_size=batch_size, num_epochs=num_epochs)  # testing

preds = trainModel(model, train_data, test_data, criterion=criterion,
                   optimizer=optimizer, batch_size=batch_size, num_epochs=num_epochs)  # training

# Get submission file
# prediction labels - preds
# test
w = open("mitricaoctavian_submission_01.txt", 'w')
w.write("id,label\n")
i = 0
for id in test["id"]:
    w.write(f"{id},{preds[i]}\n")
    i += 1

w.close()


# CNN try - SECOND SUBMISSION
# With this try I added more preprocessing, as it can be seen below
class ImagesDataset(Dataset):

    def __init__(self, df, train_flag):
        '''Create the class constructor
        df: the train/valid/test dataframes
        train_flag: True if train or valid data, False otherwise
        '''

        self.df = df
        self.train_flag = train_flag
        # Albumentations
        if self.train_flag:
            self.transform = albumentations.Compose([
                # Firstly i modified the probability
                albumentations.HorizontalFlip(p=0.7),
                albumentations.VerticalFlip(p=0.7),
                albumentations.Rotate(p=0.7),
                # Added some noise
                albumentations.GaussNoise(p=0.4),
                # And some Embossing
                albumentations.Emboss(p=0.5),
                # And some other random processing
                albumentations.HueSaturationValue(p=0.5),
                albumentations.ChannelShuffle(),
                albumentations.Normalize()
            ])
        else:
            # Albumentations are applied only during training/validation
            self.transform = albumentations.Compose([
                albumentations.Normalize()
            ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):

        # Retrieve data from dataframe
        rows = self.df.iloc[index]

        # Read image and transform
        image = cv2.imread(rows["path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed_img = self.transform(image=image)['image'].astype(np.float32)
        image = transformed_img.transpose(2, 0, 1)
        image = torch.tensor(image)

        if self.train_flag:
            # Return the image + it's label only for train/validation
            label = torch.tensor(list(map(int, rows["labels"])))
            return image, label

        else:
            return image


# run

model2 = CNN_DeepHallucination_Classifier()
model2.to(device)

# Again i loaded the model from before
checkpoint = torch.load("../input/cnn-saves/Final_Epoch_9_part2.pt")
model2.load_state_dict(checkpoint)

# Tweaked some of the hyperparameters
learning_rate = 0.002  # how much we update our weights and biases based on the error
weight_decay = 0.0008   # dial back the large weights
batch_size = 400
num_epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model2.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Adam optimizer function

testModel(model2, example_data, test_example, criterion=criterion,
          optimizer=optimizer, batch_size=batch_size, num_epochs=num_epochs)  # testing

preds = trainModel(model2, train_data, test_data, criterion=criterion,
                   optimizer=optimizer, batch_size=batch_size, num_epochs=num_epochs)  # training

# Get submission file
# prediction labels - preds
# test
w = open("mitricaoctavian_submission_02.txt", 'w')
w.write("id,label\n")
i = 0
for id in test["id"]:
    w.write(f"{id},{preds[i]}\n")
    i += 1

w.close()


# ResNet18 Try - THIRD SUBMISSION

# This one was pretty straight forward as I
# already had the auxiliary functions ready to go
model = models.resnet18(pretrained=True)
# Set the features
features = model.fc.in_features

# Adjust the linear layer with our number of classes
model.fc = nn.Linear(features, 7)
model.to(device)

# Again with the loading
checkpoint = torch.load("../input/transferlearningdeephallucination-saves/Final_Epoch_9_part2.pt")
model.load_state_dict(checkpoint)

# Hyperparameters and functions used in last run
learning_rate = 0.001
weight_decay = 0.0005
batch_size = 200
num_epochs = 40
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# I also aded a learning rate scheduler
# This function lowers the learning rate after step_size epochs by gamma
# I just added step_lr.step() after each epoch to make this work
step_lr = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

testModel(model, example_data, test_example, criterion=criterion,
          optimizer=optimizer, batch_size=batch_size, num_epochs=num_epochs)  # testing


preds = trainModel(model, example_data, test_example, criterion=criterion,
                   optimizer=optimizer, batch_size=batch_size, num_epochs=num_epochs)  # training

# Get submission file

# prediction labels - preds
# test
w = open("mitricaoctavian_submission_03.txt", 'w')
w.write("id,label\n")
i = 0
for id in test["id"]:
    w.write(f"{id},{preds[i]}\n")
    i += 1

w.close()