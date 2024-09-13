import torch  # type: ignore


def custom_normalize(tensor, dim=0, default_constant=0.5):
    """
    dim is the dimension on which to normalize
    default_constant is the value to use when the sum is zero
    """
    # Compute the sum along dim=dim and keepdim=True to maintain the dimensions for broadcasting
    sum_tensor = torch.sum(tensor, dim=dim, keepdim=True)

    # Create a mask where the sum is zero
    mask = sum_tensor == 0

    # Replace zero sums with ones to avoid division by zero and then divide
    result = tensor / torch.where(mask, torch.ones_like(sum_tensor), sum_tensor)

    # Where the sum was zero, replace with the constant C
    result = torch.where(
        mask, torch.full_like(result, fill_value=default_constant), result
    )

    return result
