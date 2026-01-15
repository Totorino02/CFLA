from torch.onnx.symbolic_opset9 import detach
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np

class Client:
    """
    This is the base class for a client.
    For each type of algorithm, we would inherit from this class and implement methods
    """
    def __init__(self, client_id, dataset, args, model=None, **kwargs):
        self.client_id = client_id
        self.dataset = dataset
        self.optimizer =  args["optimizer"]
        self.criterion = args["criterion"]
        self.device = args["device"]
        self.local_epochs = args["local_epochs"]
        self.local_batch_size = args["batch_size"]
        self.learning_rate = args["learning_rate"]
        self.local_model = model
        self.train_history = []
        self.test_history = []
        self.accuracies = []
        train_size = int(len(dataset) * args["train_fraction"])
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        self.train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=True)

    def train(self, global_model=None, verbose=False, **kwargs):
        # Save initial global model parameters
        initial_params = []
        for param in global_model.parameters():
            initial_params.append(param.flatten().detach().clone())
        initial_params = torch.cat(initial_params)
        
        # copy of the global model to a local model
        self.local_model = type(global_model)().to(self.device)
        self.local_model.load_state_dict(global_model.state_dict())
        self.local_model.train()
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean') # torch.nn.MSELoss()

        training_loss = list()
        loss = 0.0
        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                training_loss.append(loss.item())
                if verbose :# and batch_idx % 10 == 0:
                    print(f"Client {self.client_id}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
            acc, _ = self.test()
            self.accuracies.append(acc)
            self.train_history.append(loss.item())
        
        # Calculate updates (differences)
        updated_params = []
        for param in self.local_model.parameters():
            updated_params.append(param.flatten().detach())
        updated_params = torch.cat(updated_params)
        
        # Return the difference (update)
        update = updated_params - initial_params

        #self.train_history.append(training_loss)

        return update, np.mean(training_loss)

    def test(self):
        self.local_model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.local_model(data)
                test_loss = self.criterion(output, target)
                self.test_history.append(test_loss.item())
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(self.test_loader.dataset)
        return accuracy, self.test_history

    def predict(self, data):
        self.local_model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            output = self.local_model(data)
            pred = output.argmax(dim=1, keepdim=True)
        return pred.cpu().numpy()
        

    def get_params(self):
        pass

    def set_params(self, **kwargs):
        pass
