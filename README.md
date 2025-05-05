# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset
The task is to automatically classify red blood cell images into two categories: parasitized (malaria-infected) and uninfected (healthy). Malaria-infected cells contain the Plasmodium parasite, while uninfected cells are healthy. The goal is to build a convolutional neural network (CNN) to accurately distinguish between these classes.
Manual inspection of blood smears is time-consuming and prone to errors. By using deep learning, we can automate the process, speeding up diagnosis, reducing healthcare professionals' workload, and improving detection accuracy.
The dataset consists of 27,558 annotated cell images, evenly split between parasitized and uninfected cells, providing a reliable foundation for model training and testing.
### Neural Network Model
![EX-04-DL-OUTPUT (5)](https://github.com/user-attachments/assets/5d23b1b2-a0a7-4bfb-9c03-c2a768878aca)

### Design Steps
1. **Import Libraries**:Import TensorFlow, data preprocessing tools, and visualization libraries.
2. **Configure GPU**:Set up TensorFlow for GPU acceleration to speed up training.
3. **Data Augmentation**:Create an image generator for rotating, shifting, rescaling, and flipping to enhance model generalization.
4. **Build CNN Model**:Design a convolutional neural network with convolutional layers, max-pooling, and fully connected layers; compile the model.
5. **Train Model**:Split the dataset into training and testing sets, then train the model using the training data.
6. **Evaluate Performance**:Assess the model using the testing data, generating a classification report and confusion matrix.

## PROGRAM

### Name: Yogeshvar

### Register Number: 212222230074
```python
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, patience=3):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    trigger_times = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            X, y = X.to(device), y.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device).float().unsqueeze(1)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
        val_loss /= len(test_loader)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), 'best_model_resnet.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping triggered.')
                break

    print('Name: Yogeshvar')
    print('Register Number: 212222230180')
```
```python
def evaluate_model(model, test_loader, test_data):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device).float().unsqueeze(1)  # Ensure float for binary targets
            output = model(X)
            predicted = (output > 0.5).int()  # Binary prediction
            total += y.size(0)
            correct += (predicted == y.int()).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = test_data.classes  # ['parasitized', 'uninfected']

    print('Name:Yogeshvar')
    print('Register Number:212222230180')
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    print('Name:Yogeshvar')
    print('Register Number:212222230180')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/1a37a9e6-55b3-444a-94f5-dae52684b247)


### Classification Report

![image](https://github.com/user-attachments/assets/70098931-009d-4b41-8733-23cd2ddb9ebf)


### Confusion Matrix

![image](https://github.com/user-attachments/assets/02d9cfcf-df21-4ac7-bb88-7f552d424567)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/2f3f5180-fa19-417a-afeb-c73e259b1092)

## RESULT
Thus a deep neural network for Malaria infected cell recognition and to analyze the performance is created using tensorflow.
