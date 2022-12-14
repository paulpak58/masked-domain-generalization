from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms

# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="iwildcam", root_dir="data", download=True)

# # Get the training set
# train_data = dataset.get_subset(
#     "train",
#     transform=transforms.Compose(
#         [transforms.Resize((448, 448)), transforms.ToTensor()]
#     ),
# )

# # Prepare the standard data loader
# train_loader = get_train_loader("standard", train_data, batch_size=16)

# # (Optional) Load unlabeled data
# dataset = get_dataset(dataset="iwildcam", download=True, unlabeled=True)
# unlabeled_data = dataset.get_subset(
#     "test_unlabeled",
#     transform=transforms.Compose(
#         [transforms.Resize((448, 448)), transforms.ToTensor()]
#     ),
# )
# unlabeled_loader = get_train_loader("standard", unlabeled_data, batch_size=16)
