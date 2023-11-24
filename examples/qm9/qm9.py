import os, json

import numpy as np

import torch
import torch_geometric
import torch_geometric.transforms as T
import flatten_json
import wandb

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import hydragnn


class CormorantFeatures(T.BaseTransform):
    def __init__(self, max_atomic_number, max_exponent):
        self.max_atomic_number = max_atomic_number
        self.max_exponent = max_exponent

    def __call__(self, data):
        atomic_numbers = data.x[:, 5]
        exponents = torch.arange(self.max_exponent + 1).to(data.x)
        Z_vec = (
            atomic_numbers[:, None] / self.max_atomic_number
        ).pow(exponents).view(-1, exponents.size(0))
        onehots = data.x[:, :5]
        data.x = (
            onehots[..., None] * Z_vec[:, None]
        ).view(-1, 5 * (self.max_exponent + 1))

        return data


class SelectProperty(T.BaseTransform):
    def __init__(self, property_idx):
        self.property_idx = property_idx

    def __call__(self, data):
        data.y = data.y[:, self.property_idx]

        return data


# Update each sample prior to loading.
def qm9_pre_transform(data):
    # Set descriptor as element type.
    data.x = data.z.float().view(-1, 1)
    # Only predict free energy (index 10 of 19 properties) for this run.
    data.y = data.y[:, 10] / len(data.x)
    graph_features_dim = [1]
    node_feature_dim = [1]
    return data


def qm9_pre_filter(data):
    return data.idx < num_samples


# Set this path for output.
try:
    os.environ["SERIALIZED_DATA_PATH"]
except:
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

num_samples = 1000

# Configurable run choices (JSON file that accompanies this example script).
filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm9.json")
with open(filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]

# Always initialize for multi-rank training.
world_size, world_rank = hydragnn.utils.setup_ddp()

log_name = "qm9_test"
# Enable print to log file.
hydragnn.utils.setup_log(log_name)

# Use built-in torch_geometric dataset.
# Filter function above used to run quick example.
# NOTE: data is moved to the device in the pre-transform.
# NOTE: transforms/filters will NOT be re-run unless the qm9/processed/ directory is removed.
dataset = torch_geometric.datasets.QM9(
    root="dataset/qm9", # pre_transform=qm9_pre_transform, pre_filter=qm9_pre_filter
)
# EGNN splits
dataset_rng = np.random.RandomState(seed=0)
permutation = dataset_rng.permutation(len(dataset))
dataset = dataset[permutation]
splits = np.split(permutation, [100_000, 100_000 + 17_748])
max_atomic_number = dataset[splits[0]].data.x[:, 5].max().item()
dataset.transform = T.Compose([SelectProperty(1),
                               CormorantFeatures(max_atomic_number, 2)])
(train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
    *map(dataset.__getitem__, splits), config["NeuralNetwork"]["Training"]["batch_size"]
)

config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)

model = hydragnn.models.create_model_config(
    config=config["NeuralNetwork"],
    verbosity=verbosity,
)
breakpoint()
model = hydragnn.utils.get_distributed_model(model, verbosity)

learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, config["NeuralNetwork"]["Training"]["num_epoch"]
)

# Run training with the given model and qm9 dataset.
writer = hydragnn.utils.get_summary_writer(log_name)
hydragnn.utils.save_config(config, log_name)

hydragnn.train.train_validate_test(
    model,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    writer,
    scheduler,
    config["NeuralNetwork"],
    log_name,
    verbosity,
)
