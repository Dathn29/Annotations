import argparse
import torch
import os
from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import segmentation_models_pytorch as smp
from Data.dataloader import Dataset
from Data.albumentation import get_training_augmentation,get_validation_augmentation,get_preprocessing
import matplotlib.pyplot as plt



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

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
def main(opt):
  best_model = torch.load(opt.model)
  DATA_DIR =os.getcwd()
  x_test_dir = os.path.join(DATA_DIR, 'detect')
  preprocessing_fn = smp.encoders.get_preprocessing_fn(opt.encoder,opt.weight)  

  loss = smp.utils.losses.JaccardLoss()
  metrics = [smp.utils.metrics.IoU(threshold=opt.threshold),]
  test_car = Dataset(
    x_test_dir,
    x_test_dir,
    preprocessing=get_preprocessing(preprocessing_fn),
  )
  test_car_vis = Dataset(
    x_test_dir,
    x_test_dir,
  )
  for i in range(len(test_car)):
    image_vis,ma = test_car_vis[i]
    image,gt = test_car[i]
    x_tensor1 = torch.from_numpy(image).to(opt.device).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor1)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
    visualize(
        image=image_vis, 
        predicted_mask=pr_mask
    )

  
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
