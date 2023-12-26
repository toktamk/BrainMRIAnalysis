class Train_Evaluate_Model:
    def __init__(model,device,optimizer,criterion):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        return
    
    def train(train_loader):
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        device = self.device
        model.train()
        torch.autograd.set_detect_anomaly(True)
        for data1 in train_loader:
            data = data1[0]
            labels = data1[1]
            labels = labels.float().detach()
            outputs = labels
            optimizer.zero_grad()
            loss = 0
            for i in range(len(data)):
                data_x = data[i].x.detach().requires_grad_(True)
                data_edge_index = data[i].edge_index.detach()
                outputs[i] = model(data_x, data_edge_index)
                outputs[i] = outputs[i].detach().requires_grad_(True)
                loss += criterion(outputs[i].float(), labels[i].float())/len(data)
        
            loss.backward()
            optimizer.step()
        return

    def evaluate(loader):
        model = self.model
        device = self.device
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data1 in loader:
                #data1 = data1.to(device)
                data = data1[0]
                labels = data1[1]
                for i in range(len(data)):
                    data_x = data[i].x.detach()
                    data_edge_index = data[i].edge_index.detach()
                    label = labels[i]
                    output = model(data_x, data_edge_index)
                    _, predicted = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
        accuracy = 100 * correct / total
        return accuracy