import argparse
import torch
import os
from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import segmentation_models_pytorch as smp
from Data.dataloader import Dataset
from Data.albumentation import get_training_augmentation,get_validation_augmentation,get_preprocessing




def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='resnet34')
    parser.add_argument('--weight', type=str, default='imagenet')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threshold', default=0.5, help='vary img-size +/- 50%%')
    parser.add_argument('--saved', type=str, default='', help='version of dataset artifact to be used')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
  best_model = torch.load(opt.model)
  DATA_DIR =os.getcwd()
  x_valid_dir = os.path.join(DATA_DIR, 'data3/val/images')
  y_valid_dir = os.path.join(DATA_DIR, 'data3/val/masks')
  preprocessing_fn = smp.encoders.get_preprocessing_fn(opt.encoder,opt.weight)  

  test_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir,
    preprocessing=get_preprocessing(preprocessing_fn),
  )
  loss = smp.utils.losses.JaccardLoss()
  metrics = [smp.utils.metrics.IoU(threshold=opt.threshold),]
  
  test_dataloader = DataLoader(test_dataset)
  test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device= opt.device,
  )

  logs = test_epoch.run(test_dataloader)
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
