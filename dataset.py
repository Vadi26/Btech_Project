
import torch
from torch.utils.data import random_split, Dataset, DataLoader, TensorDataset
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import numpy as np

def get_mnist(data_path: str = './data'):

    # Normalize ((Variance,), (stddev,))
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset

def prepare_dataset(num_partitions: int, 
                    batch_size: int,
                    randomness: bool,
                    val_ratio: float=0.1):
    
    if randomness==False:
        # Downloads the dataset
        trainset, testset = get_mnist()

        # Split trainset into 'num_partitions' trainsets so that we can assign a portion of the dataset to every client
        num_images = len(trainset) // num_partitions
        partition_len = [num_images] * num_partitions
        trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))

        # Create dataloaders with train + val support
        # We split the created trainsets into trainset and valset
        trainloaders = []
        valloaders = []
        for trainset_ in trainsets:
            num_total = len(trainset_)
            num_val = int(val_ratio * num_total)
            num_train = num_total - num_val

            for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

            trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
            valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))
        
        testloader = DataLoader(testset, batch_size=128)

    else:
        # Downloads the dataset
        trainset, testset = get_mnist()

        # Total size of the dataset
        total_size = len(trainset)

        # Generate random partition sizes such that their sum equals total_size
        random_sizes = np.random.dirichlet(np.ones(num_partitions)) * total_size
        partition_len = [int(size) for size in random_sizes]

        # Adjust the last partition to ensure all samples are allocated
        partition_len[-1] += total_size - sum(partition_len)

        # Ensure the partition lengths sum up to the total dataset size
        assert sum(partition_len) == total_size, "Partition lengths must sum up to the total dataset size."

        # Split trainset into random-sized partitions
        trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))

        # Create dataloaders with train + val support
        trainloaders = []
        valloaders = []
        for trainset_ in trainsets:
            num_total = len(trainset_)
            num_val = int(val_ratio * num_total)
            num_train = num_total - num_val

            for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

            trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
            valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))

        testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader

class CreditCardDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

def get_credit_card_data(data_path: str):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Separate features and labels
    X = df.drop(columns=['Class', 'id'])  # Drop Class and id columns
    y = df['Class']

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Ensure compatibility with PyTorch by converting labels to numpy arrays
    y_train = y_train.values
    y_test = y_test.values

    # Create PyTorch datasets
    train_dataset = CreditCardDataset(X_train, y_train)
    test_dataset = CreditCardDataset(X_test, y_test)

    return train_dataset, test_dataset

def prepare_credit_card_dataset(data_path: str,
                                num_partitions: int, 
                                batch_size: int,
                                val_ratio: float=0.1):
    # Load the credit card fraud detection dataset
    trainset, testset = get_credit_card_data(data_path=data_path)

    # Total size of the dataset
    total_size = len(trainset)

    # Generate random partition sizes such that their sum equals total_size
    random_sizes = np.random.dirichlet(np.ones(num_partitions)) * total_size
    partition_len = [int(size) for size in random_sizes]

    # Adjust the last partition to ensure all samples are allocated
    partition_len[-1] += total_size - sum(partition_len)

    # Ensure the partition lengths sum up to the total dataset size
    assert sum(partition_len) == total_size, "Partition lengths must sum up to the total dataset size."

    # Split the dataset into random-sized partitions
    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))

    # Create dataloaders with train + validation support
    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))

    # Create a dataloader for the test set
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloaders, valloaders, testloader

def outliers_dropping_based_in_3sigma(dataframe, numeric_cols_to_trim):

    for column in numeric_cols_to_trim:
        column_mean = dataframe[column].mean()
        column_3_sigma = 3*dataframe[column].std()

        dataframe = dataframe[(dataframe[column]) < (column_mean + column_3_sigma)]

    return dataframe

def no_underscore_labels_eraser(dataframe):
  return dataframe[(dataframe["label"].str.find("-") != -1) &
                   (dataframe["label"].str.find("-") != -1) |
                   (dataframe["label"] == "BenignTraffic")]

def label_transformer(label: str):
  if label == "BenignTraffic":
    return label

  character = "-" if "-" in label else "_"

  return label.split(character)[0]

def label_selection(dataframe):
  no_underscore_df = no_underscore_labels_eraser(dataframe)
  label_transformer_vec = np.vectorize(label_transformer)
  return label_transformer_vec(no_underscore_df["label"])

def get_iot_data(data_path: str):
    # Load the dataset
    df = pd.read_csv(data_path)

    df = df.dropna()

    valid_categorical_data = ['fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 'HTTP', 'HTTPS', 'TCP', 'UDP', 'ICMP']
    valid_numeric_columns_for_standardization = ['flow_duration', 'Header_Length', 'Duration', 'Rate', 'Srate', 'Drate', 'fin_count', 'urg_count', 'rst_count', 'Max', 'Covariance']

    prueba_df = outliers_dropping_based_in_3sigma(df, valid_numeric_columns_for_standardization)

    all_labels = pd.unique(df["label"])

    tmp_counter = 0
    no_underscore_columns = []

    for info in all_labels:

        if info.find("BenignTraffic") != -1:
            continue

        if info.find("-") == -1 and info.find("_") == -1:
            tmp_counter += 1
            no_underscore_columns.append(info)

    selected_lables = label_selection(df)

    df_with_selected_labels = no_underscore_labels_eraser(df)
    df_with_selected_labels.loc[:,"final_label"] = selected_lables

    final_columns = valid_numeric_columns_for_standardization + valid_categorical_data + ["final_label"]
    
    X, y = df_with_selected_labels[final_columns[:-1]], df_with_selected_labels[final_columns[-1]]

    label_enco = LabelEncoder()
    y = label_enco.fit_transform(y.values.reshape(-1, 1))
    class_names = label_enco.classes_

    for col in valid_categorical_data :
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Ensure compatibility with PyTorch by converting labels to numpy arrays
    # y_train = y_train
    # y_test = y_test
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)


    # Create PyTorch datasets
    train_dataset = CreditCardDataset(X_train, y_train)
    test_dataset = CreditCardDataset(X_test, y_test)

    return train_dataset, test_dataset

# Same attacck partition
# def prepare_iot_dataset(data_path: str,
#                         num_partitions: int, 
#                         batch_size: int,
#                         val_ratio: float=0.1):
#     # Load the IoT intrusion detection dataset
#     trainset, testset = get_iot_data(data_path=data_path)

#     # Group data by attack type (final_label)
#     attack_types = {label: [] for label in range(num_partitions)}  # Assuming labels are 0-5
#     for idx in range(len(trainset)):
#         data, label = trainset[idx]  # Extract features and label
#         attack_types[int(label.item())].append((data, int(label.item())))  # Store in the respective partition

#     # Ensure each client gets data from only one attack type
#     trainsets = []
#     for attack_label in attack_types:
#         dataset = torch.utils.data.TensorDataset(
#             torch.stack([x[0] for x in attack_types[attack_label]]),  # Stack features
#             torch.tensor([x[1] for x in attack_types[attack_label]], dtype=torch.long)  # Labels as long type
#         )
#         trainsets.append(dataset)

#     # Create dataloaders with train + validation support
#     trainloaders = []
#     valloaders = []
#     for trainset_ in trainsets:
#         num_total = len(trainset_)
#         num_val = int(val_ratio * num_total)
#         num_train = num_total - num_val

#         for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

#         trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
#         valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))

#     # Create a dataloader for the test set
#     testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

#     return trainloaders, valloaders, testloader


# Random partition
def prepare_iot_dataset(data_path: str,
                                num_partitions: int, 
                                batch_size: int,
                                val_ratio: float=0.1):
    # Load the credit card fraud detection dataset
    trainset, testset = get_iot_data(data_path=data_path)

    # Total size of the dataset
    total_size = len(trainset)

    # Generate random partition sizes such that their sum equals total_size
    random_sizes = np.random.dirichlet(np.ones(num_partitions)) * total_size
    partition_len = [int(size) for size in random_sizes]

    # Adjust the last partition to ensure all samples are allocated
    partition_len[-1] += total_size - sum(partition_len)

    # Ensure the partition lengths sum up to the total dataset size
    assert sum(partition_len) == total_size, "Partition lengths must sum up to the total dataset size."

    # Split the dataset into random-sized partitions
    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))

    # Create dataloaders with train + validation support
    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))

    # Create a dataloader for the test set
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloaders, valloaders, testloader

# Working code for equal no.of partitions
# def prepare_credit_card_dataset(data_path: str,
#                                 num_partitions: int, 
#                                 batch_size: int,
#                                 val_ratio: float=0.1):
#     # Load the credit card fraud detection dataset
#     trainset, testset = get_credit_card_data(data_path=data_path)

#     # Split trainset into 'num_partitions' trainsets to assign a portion of the dataset to each client
#     total_size = len(trainset)
#     partition_len = [total_size // num_partitions] * num_partitions
#     partition_len[-1] += total_size % num_partitions  # Add the remainder to the last partition

#     # Ensure the partition lengths sum up to the total dataset size
#     assert sum(partition_len) == total_size, "Partition lengths must sum up to the total dataset size."

#     trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))

#     # Create dataloaders with train + validation support
#     trainloaders = []
#     valloaders = []
#     for trainset_ in trainsets:
#         num_total = len(trainset_)
#         num_val = int(val_ratio * num_total)
#         num_train = num_total - num_val

#         for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

#         trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
#         valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))
    
#     # Create a dataloader for the test set
#     testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

#     return trainloaders, valloaders, testloader

