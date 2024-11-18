import torch
import matplotlib.pyplot as plt


def plot_tensor_image(tensor: torch.Tensor):
    """
    Plots a PyTorch tensor of shape [3, h, w] using matplotlib without a frame.

    Args:
        tensor (torch.Tensor): A PyTorch tensor of shape [3, h, w].
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Tensor must have shape [3, h, w].")

    # Convert the tensor to a numpy array and transpose to [h, w, 3]
    image = tensor.permute(1, 2, 0).cpu().numpy()

    # # Ensure values are in range [0, 1] for proper visualization
    # if image.max() > 1.0:
    #     image = image / 255.0

    # Plot the image
    plt.imshow(image)
    plt.axis('off')  # Remove axes
    plt.gca().set_frame_on(False)  # Remove the frame
    plt.show()
