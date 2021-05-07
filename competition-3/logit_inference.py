import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import *
from tqdm import tqdm
from model import *

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# @torch.no_grad()
def inference(dataset_path, model_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # model = smp.FPN(
    #     encoder_name = 'timm-efficientnet-b2',
    #     encoder_weights = 'noisy-student',
    #     classes = 12)
    
    if args.predtype not in [5, 6]:
        raise("wrong predtype")
    
    model = UNet_3Plus_DeepSup_CGM(n_classes=12)
    model_path = os.path.join(model_dir, 'best5.pth' if args.predtype==5 else 'best6.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)

    model.eval()
    
    test_transform = A.Compose([
                            ToTensorV2()
                            ])

    test_path = dataset_path + '/test.json'
    dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers = 4,
        pin_memory=use_cuda,
        drop_last=False,
        collate_fn = collate_fn
    )

    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    preds_array = np.empty((0, size*size), dtype=np.long)
    with torch.no_grad():
       for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):

            # inference (512 x 512)
            if args.predtype==5:
                outs = model(torch.stack(imgs).float().to(device))
                outs = (outs[0] + outs[1] + outs[2] + outs[3] + outs[4])/5
            else:
                outs = model(torch.stack(imgs).float().to(device))
                outs = (outs[0] + outs[1] + outs[2] + outs[3] + outs[4])/4
            
            if step == 0:
                all_masks=outs.detach().cpu().numpy()
                continue
            all_masks = np.concatenate([all_masks, outs.detach().cpu().numpy()])
    
    np.save(model_dir+'all_masks.npy',all_masks)    
    print("End prediction.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for validing (default: 16)')
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    
    parser.add_argument('--predtype', type=int, default=5)

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    inference(data_dir, model_dir,  args)

    