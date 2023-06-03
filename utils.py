import torch

def catch_nan(**kwargs):
    nan_detected = False
    inf_detected = False
    for k, v in kwargs.items():
        if v.isnan().any():
            print(f'{k} is NaN')
            nan_detected = True
        if v.isinf().any():
            print(f'{k} is inf')
            inf_detected = True            
        
    if nan_detected != False:
        for k, v in kwargs.items():
            torch.set_printoptions(precision=25)
            torch.save(v, f'log/{k}.pt')
        raise ValueError('NaN detected')
    if inf_detected != False:
        for k, v in kwargs.items():
            torch.set_printoptions(precision=25)
            torch.save(v, f'log/{k}.pt')
        raise ValueError('infinity detected')

def save_tensors(**kwargs):
    for k, v in kwargs.items():
        torch.set_printoptions(precision=25)
        torch.save(v, f'log/{k}.pt')
