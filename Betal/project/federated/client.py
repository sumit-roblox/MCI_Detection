"""
Flower Federated Learning Client.
Runs local training and communicates with the federated server.
"""
import flwr as fl
import torch
from collections import OrderedDict

class MultimodalClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Call local train function here
        return self.get_parameters(config=None), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # Call local evaluation function here
        loss = 0.0 # Placeholder
        accuracy = 0.0 # Placeholder
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}

def start_client():
    # Initialize model, data, and start Flower client
    # fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=MultimodalClient(model, trainloader, valloader))
    pass

if __name__ == "__main__":
    start_client()
