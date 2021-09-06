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
    parser.add_argument('--activation', type=str, default='sigmoid')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threshold', default=0.5, help='vary img-size +/- 50%%')
    parser.add_argument('--saved', type=str, default='', help='version of dataset artifact to be used')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
  DATA_DIR =os.getcwd()
  x_train_dir = os.path.join(DATA_DIR, 'data3/train/images')
  y_train_dir = os.path.join(DATA_DIR, 'data3/train/masks')

  x_valid_dir = os.path.join(DATA_DIR, 'data3/val/images')
  y_valid_dir = os.path.join(DATA_DIR, 'data3/val/masks')

  ENCODER = opt.encoder
  ENCODER_WEIGHTS = opt.weight
  ACTIVATION = opt.activation 
  DEVICE = opt.device

  model = smp.Unet(
      encoder_name=ENCODER, 
      encoder_weights=ENCODER_WEIGHTS, 
      classes=1, 
      activation=ACTIVATION,
  )
  preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)  
  train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
  )

  valid_dataset = Dataset(
      x_valid_dir, 
      y_valid_dir, 
      augmentation=get_validation_augmentation(), 
      preprocessing=get_preprocessing(preprocessing_fn),
  )
  train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=12)
  valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)
  loss = smp.utils.losses.JaccardLoss()
  metrics = [smp.utils.metrics.IoU(threshold=opt.threshold),]
  optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001),])
  
  train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
  )

  valid_epoch = smp.utils.train.ValidEpoch(
      model, 
      loss=loss, 
      metrics=metrics, 
      device=DEVICE,
      verbose=True,
  )
  max_score = 0
  for i in range(0, opt.epochs):
      
      print('\nEpoch: {}'.format(i))
      train_logs = train_epoch.run(train_loader)
      valid_logs = valid_epoch.run(valid_loader)
      
      # do something (save model, change lr, etc.)
      if max_score < valid_logs['iou_score']:
          max_score = valid_logs['iou_score']
          torch.save(model, opt.saved)
          print('Model saved!') 
      if i == 25:
          optimizer.param_groups[0]['lr'] = 1e-5
          print('Decrease decoder learning rate to 1e-5!')
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
