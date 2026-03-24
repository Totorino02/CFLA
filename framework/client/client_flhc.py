import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from framework.client.clientbase import Client
from torch.nn import Module
import os
from declearn.main.utils._energy_monitor import EnergyMonitor # type: ignore


RAPL_ENERGY_UNITS = 1e6
NVML_NVIDIA_UNITS = 1e3

class ClientFLHC(Client):

    def __init__(self, client_id, dataset, args, output_dir, **kwargs):
        super().__init__(client_id, dataset, args, **kwargs)
        self.local_model : Module = None
        self.device = args["device"]
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean') # args["criterion"]
        self.optimizer = args["optimizer"]
        self.local_epochs = args["local_epochs"]
        self.learning_rate = args["learning_rate"]
        self.history = []
        self.output_dir = output_dir
        train_size = int(len(dataset) * args["train_fraction"])
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        self.train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=True)

        if not os.path.exists(os.path.join(self.output_dir, f"client_{self.client_id}")):
            os.makedirs(os.path.join(self.output_dir, f"client_{self.client_id}"))
        
        # create metrics.csv file
        with open(os.path.join(self.output_dir, f"client_{self.client_id}", "metrics.csv"), "w") as f:
            f.write("round,loss,accuracy_before, accuracy_after, energy_consumed, energy_ratio\n")

    def train(self, global_model=None, verbose=False, save_metrics=True, **kwargs):
        """
        This method trains the local model on the local dataset
        and returns the updated vector (the difference between the global model params and the local trained model params)
         and the last training loss.
        :param global_model: The global model gets from the server
        :param verbose: If True, prints the training progress
        :return: Update_vector, loss
        """
        # copy of the global model to a local model
        self.local_model = type(global_model)().to(self.device)
        self.local_model.load_state_dict(global_model.state_dict())
        self.local_model.train()
        self.optimizer = torch.optim.SGD(params=self.local_model.parameters(), lr=self.learning_rate)

        # training on local data
        loss = torch.tensor(0.0)

        acc_before, _, _ = self.evaluate()
        
        # Start energy monitoring
        energy_monitor = EnergyMonitor()
        energy_monitor.start()


        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        
        # Stop energy monitoring and get results
        energy_consumed = energy_monitor.stop()


        # update vector of local model parameters
        update_vector = []
        updated_params = []
        for new_param, old_param in zip(self.local_model.parameters(), global_model.parameters()):
            update_vector.append((new_param.data - old_param.data).flatten())
            updated_params.append(new_param.data.flatten())

        
        # updates_norm
        client_energy = 0 #energy_monitor.compute_energy_ratio(energy_consumed, self.local_model.parameters())
        udpates_norm = torch.norm(torch.cat(update_vector))
        for k, e in energy_consumed.items():
            if e < 0:
                e = 0
            if k.startswith("nvidia"):
                e = e / NVML_NVIDIA_UNITS
            else:
                e = e / RAPL_ENERGY_UNITS
            client_energy += e 
        energy_ratio = client_energy / (udpates_norm.item() + 1e-9)


        # evaluate the local model
        accuracy, _, _ = self.evaluate()

        if save_metrics:
            # Save metrics to CSV
            with open(os.path.join(self.output_dir, f"client_{self.client_id}", "metrics.csv"), "a") as f:
                f.write(f"{kwargs.get('round', 0)},{loss.item()},{acc_before},{accuracy},{energy_consumed['package_0']},{energy_ratio}\n")
        
        if verbose:
                print(f"Client: {self.client_id} | Round: {kwargs.get('round', 0)} | Loss: {loss.item()} | Acc: {accuracy} | Engy: {energy_consumed} | E. Ratio: {energy_ratio}")

        #for vect in update_vector:
        #    print(vect.size())
        update_vector = torch.cat(update_vector).numpy()
        updated_params = torch.cat(updated_params)
        return updated_params, update_vector, loss.item()

    def evaluate(self):
        self.local_model.eval()
        correct = 0
        total = 0
        test_loss = torch.tensor(0.0)
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.local_model(data)
                test_loss = self.criterion(output, target)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = correct / total
        return accuracy, correct, test_loss.item()

    def predict(self):
        pass

    def get_params(self):
        return self.local_model.state_dict()

    def set_params(self, model_params: dict):
        self.local_model.load_state_dict(model_params)
