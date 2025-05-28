import argparse
from os import makedirs
from os.path import join
import itertools
import yaml

from model import BWHandGestureRecognitionModel

# parses line args
parser = argparse.ArgumentParser(
    prog="Leap2 Model", description="Generate configs for the pipeline"
)
parser.add_argument(
    "--path", default=join(".", "cfgs"), help="Where to save the configs"
)
parser.add_argument(
    "--datasets_path",
    default=join("..", "..", "datasets"),
    help="Where the datasets are located",
)
parser.add_argument("--batch_size", default=256)
parser.add_argument("--lr", default=5e-5)
parser.add_argument("--max_epochs", default=3)
parser.add_argument("--seed", default=42)
line_args = vars(parser.parse_args())

# creates the directory
makedirs(line_args["path"], exist_ok=True)


def create_dict(
    dataset,
    datasets_path,
    device="auto",
    seed=line_args["seed"],
    validation="loso",
    use_horizontal_image=True,
    use_vertical_image=True,
    use_horizontal_landmarks=True,
    use_vertical_landmarks=True,
    image_backbone_name="clip",
    landmarks_backbone_name="mlp",
    checkpoints_path="./checkpoints",
    batch_size=line_args["batch_size"],
    max_epochs=line_args["max_epochs"],
    lr=line_args["lr"],
):
    return {
        "name": f"{dataset}_{validation}_{image_backbone_name}_{landmarks_backbone_name}",
        "dataset": dataset,
        "dataset_path": join(datasets_path, dataset),
        "device": device,
        "seed": seed,
        "validation": validation,
        "use_horizontal_image": use_horizontal_image,
        "use_vertical_image": use_vertical_image,
        "use_horizontal_landmarks": use_horizontal_landmarks,
        "use_vertical_landmarks": use_vertical_landmarks,
        "image_backbone_name": image_backbone_name,
        "landmarks_backbone_name": landmarks_backbone_name,
        "checkpoints_path": checkpoints_path,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "lr": lr,
    }


# defines the parameters
datasets = ["ml2hp"]
validations = ["loso"]

# creates the ablation configurations
makedirs(join(line_args["path"], "ablation"), exist_ok=True)
for image_backbone_name, landmarks_backbone_name in itertools.product(BWHandGestureRecognitionModel._possible_image_backbones, BWHandGestureRecognitionModel._possible_landmarks_backbones):
    cfg = create_dict(
        dataset="ml2hp",
        datasets_path=line_args["datasets_path"],
        image_backbone_name=image_backbone_name,
        landmarks_backbone_name=landmarks_backbone_name,
        checkpoints_path="./checkpoints/ablation",
        use_horizontal_image=True if image_backbone_name is not None else False,
        use_vertical_image=True if image_backbone_name is not None else False,
        use_horizontal_landmarks=True if landmarks_backbone_name is not None else False,
        use_vertical_landmarks=True if landmarks_backbone_name is not None else False,
    )
    filename = f"{cfg['name']}.yaml"
    with open(join(line_args["path"], "ablation", filename), "w") as file:
        yaml.dump(cfg, file)

# exit()
# # loops over each configuration
# for (
#     dataset,
#     validation,
#     use_horizontal_image,
#     use_vertical_image,
#     use_horizontal_landmarks,
#     use_vertical_landmarks,
# ) in itertools.product(
#     datasets, validations, [True, False], [True, False], [True, False], [True, False]
# ):
#     num_images = sum([1 if use_horizontal_image else 0, 1 if use_vertical_image else 0])
#     num_landmarks = sum(
#         [1 if use_horizontal_landmarks else 0, 1 if use_vertical_landmarks else 0]
#     )
#     if (
#         (num_images == 0 and num_landmarks == 0)
#         or (num_images == 2 and num_landmarks == 1)
#         or (num_images == 1 and num_landmarks == 2)
#     ):
#         continue
#     content = {
#         "dataset": dataset,
#         "dataset_path": join(line_args["datasets_path"], dataset),
#         "device": "auto",
#         "seed": line_args["seed"],
#         "validation": validation,
#         "use_horizontal_image": use_horizontal_image,
#         "use_vertical_image": use_vertical_image,
#         "use_horizontal_landmarks": use_horizontal_landmarks,
#         "use_vertical_landmarks": use_vertical_landmarks,
#         "checkpoints_path": "./checkpoints",
#         "batch_size": line_args["batch_size"],
#         "max_epochs": line_args["max_epochs"],
#         "lr": line_args["lr"],
#     }
#     filename = f"{dataset}_{validation}_images={'h' if use_horizontal_image else ''}{'v' if use_vertical_image else ''}_landmarks={'h' if use_horizontal_landmarks else ''}{'v' if use_vertical_landmarks else ''}.yaml"
#     with open(join(line_args["path"], filename), "w") as file:
#         yaml.dump(content, file)
