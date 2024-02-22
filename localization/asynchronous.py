import torch
import asyncio
import numpy as np

async def transfer_and_write_async(tensor_gpu, file_path):
    # Transfer tensor to CPU synchronously
    tensor_cpu = tensor_gpu.to(device='cpu')

    # Do something with the tensor on CPU (e.g., save to a file asynchronously)
    np_array = tensor_cpu.numpy()

    # Save the NumPy array to a file asynchronously
    await asyncio.to_thread(np.save, file_path, np_array)

async def main():
    # Assuming you have a PyTorch tensor on GPU
    tensor_gpu = torch.randn(1000, 1000, device='cuda')

    # Specify the file path for saving the NumPy array
    file_path = 'output.npy'

    # Launch the coroutine to transfer and write asynchronously
    transfer_and_write_task = asyncio.create_task(transfer_and_write_async(tensor_gpu, file_path))

    # Continue with other tasks in the main process

    # Do not wait for the task to complete here
    await asyncio.gather(transfer_and_write_task)
if __name__ == "__main__":
    asyncio.run(main())