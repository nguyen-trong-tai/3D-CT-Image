import torch
import numpy as np

def run_inference(model, volume: np.ndarray, device="cuda") -> np.ndarray:
    """
    Run segmentation inference.
    """
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(volume)[None, None].to(device)
        pred = model(x)
        prob = torch.sigmoid(pred).cpu().numpy()[0, 0]
    return prob
