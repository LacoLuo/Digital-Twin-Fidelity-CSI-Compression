import torch 
import datetime
from einops import rearrange

def save_model(model, store_model_path):
  now = datetime.datetime.now().strftime("%H_%M_%S")
  date = datetime.date.today().strftime("%y_%m_%d")
  comment = "_".join([now, date])
  
  torch.save(model.state_dict(), f'{store_model_path}/{comment}.ckpt')
  return

def load_model(model, load_model_path):
  print(f'Load model from {load_model_path}')
  model.load_state_dict(torch.load(f'{load_model_path}'))
  return model

def cal_nmse(A, B):
    """
    Compute the Normalized Mean Squared Error (NMSE) between matrices A and B using PyTorch.

    Args:
    - A (torch.Tensor): The original matrix.
    - B (torch.Tensor): The approximated matrix.

    Returns:
    - float: The NMSE value between matrices A and B.
    """
    
    # Calculate the Frobenius norm difference between A and B

    A = rearrange(A, 'b RealImag Nt Nc -> b Nt Nc RealImag').contiguous()
    B = rearrange(B, 'b RealImag Nt Nc -> b Nt Nc RealImag').contiguous()

    A = torch.view_as_complex(A)
    B = torch.view_as_complex(B)

    error_norm = torch.norm(A - B, p='fro', dim=(-1, -2))
    
    # Calculate the Frobenius norm of A
    A_norm = torch.norm(A, p='fro', dim=(-1, -2))
    
    # Return NMSE
    return (error_norm**2) / (A_norm**2)