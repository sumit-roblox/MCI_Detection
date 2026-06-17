"""
Flower Federated Learning Server.
Aggregates model updates using FedAvg.
"""
import flwr as fl

def start_server():
    """
    Start the federated learning server using FedAvg strategy.
    """
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )
    
    print("Starting federated server...")
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )

if __name__ == "__main__":
    start_server()
