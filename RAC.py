import torch
import cv2
import numpy as np

def RAC(pred_A: torch.Tensor, pred_B: torch.Tensor, video_seq: torch.Tensor) -> torch.Tensor:

    TARGET_H, TARGET_W = 256, 256
    MASK_THRESHOLD = 0.5
    device = video_seq.device
    dtype = video_seq.dtype


    with torch.no_grad():
        def tensor2np(tensor: torch.Tensor) -> np.ndarray:
            return tensor.squeeze(0).detach().cpu().numpy()

        def np2tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=dtype)

        def get_foreground_mask(pred_np: np.ndarray) -> np.ndarray:
            _, mask = cv2.threshold(pred_np, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
            return mask.astype(np.uint8)

        def mix_images(source_np: np.ndarray, target_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
            result = target_np.copy()
            result[mask != 0] = source_np[mask != 0]
            return result

        img_a_np = tensor2np(video_seq[0])
        img_b_np = tensor2np(video_seq[-1])

        mask_A = get_foreground_mask(tensor2np(pred_A))
        Ab_np = mix_images(img_a_np, img_b_np, mask_A)

        mask_B = get_foreground_mask(tensor2np(pred_B))
        Ba_np = mix_images(img_b_np, img_a_np, mask_B)


        Ab_tensor = np2tensor(Ab_np)
        Ba_tensor = np2tensor(Ba_np)


        enhanced_seq = torch.cat([
            Ab_tensor.unsqueeze(0),
            video_seq[1:-1],
            Ba_tensor.unsqueeze(0)
        ], dim=0)

    return enhanced_seq

if __name__ == '__main__':
    import torch
    import torchprofile
    import os


    predA = torch.randn(1, 256, 256).to(dtype=torch.float32)
    predB = torch.randn(1, 256, 256).to(dtype=torch.float32)
    video = torch.randn(10,1, 256, 256).to(dtype=torch.float32)

    mix_video = RAC(predA,predB,video)
    print(mix_video.shape)

