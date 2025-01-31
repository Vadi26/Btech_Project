
import logging
from collections import OrderedDict
from typing import Dict
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl

from hydra.utils import instantiate

from model import train, test, train_card_detection, test_card_detection, test_iot, train_iot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to terminal
        logging.FileHandler("flower_training.log")  # Log to file
    ]
)

class CreditCardFLowerClient(fl.client.NumPyClient):
    def __init__(self,
                 trainloader,
                 valloader,
                 model_cfg,
                 cid) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        # Instantiate the model using the configuration
        self.model = instantiate(model_cfg)
        self.cid = cid

        # Set device (GPU if available, else CPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def set_parameters(self, parameters):
        # Convert parameters into the model's state_dict
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        # Return the model's parameters as a list of numpy arrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # Receives parameters from the server and performs local training
    def fit(self, parameters, config):
        # Set global model parameters
        self.set_parameters(parameters)

        # Get hyperparameters from the server config
        lr = config.get("lr", 0.01)  # Default learning rate is 0.01 if not provided
        momentum = config.get("momentum", 0.9)  # Default momentum is 0.9
        epochs = config.get("local_epochs", 1)  # Default is 1 local epoch

        # Define optimizer
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # Perform local training
        train_loss = train_card_detection(self.model, self.trainloader, optim, epochs, self.device)

        logging.info(f"Client {self.cid} - Training Loss: {train_loss:.4f}")

        # Return updated parameters and training data size
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    # Receives parameters from the server and evaluates on the validation set
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # Set global model parameters
        self.set_parameters(parameters)

        # Perform evaluation on the validation set
        loss, accuracy = test_card_detection(self.model, self.valloader, self.device)

        logging.info(f"Client {self.cid} - Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

        # Save results to a file (optional)
        with open(f"client_{self.cid}_eval_results.log", "a") as f:
            f.write(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}\n")

        # Return evaluation metrics
        return float(loss), len(self.valloader.dataset), {"accuracy": accuracy}


# This function will be called by the main.py
# It generates a function for the server to spawn clients
def generate_credit_card_client_fn(trainloaders, valloaders, model_cfg):
    def client_fn(cid: str):
        """
        Instantiate a client with the corresponding trainloader and valloader
        based on the client ID (cid).
        """
        print(f"--------------------------{len(trainloaders[int(cid)].dataset)} , {len(trainloaders[int(cid)])}--------------------------")
        return CreditCardFLowerClient(trainloader=trainloaders[int(cid)],
                                      valloader=valloaders[int(cid)],
                                      model_cfg=model_cfg,
                                      cid=cid)

    return client_fn


class FLowerClient(fl.client.NumPyClient):
    def __init__(self,
                 trainloader,
                 valloader,
                 model_cfg,
                 cid) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        self.model = instantiate(model_cfg)
        self.cid = cid

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # Receives from the server the parameters of the global model and a set of instructions(like the hyperparameters)
    def fit(self, paramaters, config):
        # Copy parameters sent by the server into local model
        self.set_parameters(paramaters)

        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # Do local training
        train(self.model, self.trainloader, optim, epochs, self.device)

        return self.get_parameters({}), len(self.trainloader), {}
    
    # Receives the parameters of the global model and evaluates the using the local validation set and returns the corresponding metrics
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # Copy parameters sent by the server into local model
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)

        logging.info(f"Client {self.cid} - Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

        return float(loss), len(self.valloader), {"accuracy": accuracy}
    

# This function will be called by the main.py
# This is a function for the server to spawn clients
def generate_client_fn(trainloaders, valloaders, model_cfg):

    def client_fn(cid: str):

        return IotClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            model_cfg=model_cfg,
                            cid=cid
                            )

    return client_fn
    # return NumpyClient.to_client(client_fn)

class IotClient(fl.client.NumPyClient):
    def __init__(self,
                 trainloader,
                 valloader,
                 model_cfg,
                 cid) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        self.model = instantiate(model_cfg)
        self.cid = cid

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # Receives from the server the parameters of the global model and a set of instructions(like the hyperparameters)
    def fit(self, paramaters, config):
        # Copy parameters sent by the server into local model
        self.set_parameters(paramaters)

        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # Do local training
        train_iot(self.model, self.trainloader, optim, epochs, self.device)

        return self.get_parameters({}), len(self.trainloader), {}
    
    # Receives the parameters of the global model and evaluates the using the local validation set and returns the corresponding metrics
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # Copy parameters sent by the server into local model
        self.set_parameters(parameters)

        loss, accuracy = test_iot(self.model, self.valloader, self.device)

        server_round = config.get("server_round", "unknown")

        logging.info(f"Round {server_round} : Client {self.cid} - Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

        results = {"round": server_round, "client_id": self.cid, "loss": loss, "accuracy": accuracy}

        with open("client_information.txt", "a") as f:
            f.write(str(results) + "\n")

        return float(loss), len(self.valloader), {"accuracy": accuracy}
    

def generate_iot_client(trainloaders, valloaders, model_cfg):

    def client_fn(cid: str):

        return IotClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            model_cfg=model_cfg,
                            cid=cid
                            )

    return client_fn
    # return NumpyClient.to_client(client_fn)