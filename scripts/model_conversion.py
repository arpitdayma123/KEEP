import torch
from collections import OrderedDict


def convert_KEEP(checkpoint_path, save_path):
    """Convert checkpoint by renaming cross_fuse to cfa and fuse_convs_dict to cft.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        save_path (str): Path to save the converted checkpoint
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)['params_ema']
    
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        if 'cross_fuse' in k:
            new_k = k.replace('cross_fuse', 'cfa')
        elif 'fuse_convs_dict' in k:
            new_k = k.replace('fuse_convs_dict', 'cft')
        else:
            new_k = k
        new_state_dict[new_k] = v
        print(f'Processing key: {k} -> {new_k}')
        
    torch.save(OrderedDict({'params_ema': new_state_dict}), save_path)
    print(f'Saved converted checkpoint to {save_path}')


if __name__ == '__main__':
    checkpoint_path = '/mnt/sfs-common/rcfeng/video_face/CodeFormer/experiments/Asian_stage3_combine5000_ftold_ft120k/models/net_g_30000.pth'
    save_path = 'weights/KEEP/KEEP_Asian.pth'
    convert_KEEP(checkpoint_path, save_path)
