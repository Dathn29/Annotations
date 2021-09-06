import albumentations
def get_training_augmentation():
    train_transform = [

        albumentations.Flip(p=0.5), 
        albumentations.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albumentations.ChannelShuffle(),
        albumentations.ToGray(p=0.5),
        albumentations.RGBShift(p=0.5),
        #albu.PadIfNeeded(min_height=64, min_width=64, always_apply=True, border_mode=0),
        #albu.RandomCrop(height=64, width=64, always_apply=True),

        albumentations.IAAPerspective(p=0.5),
    ]
    return albumentations.Compose(train_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_validation_augmentation():
    test_transform = [
        albumentations.PadIfNeeded(64, 64)
    ]
    return albumentations.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    _transform = [
        albumentations.Lambda(image=preprocessing_fn),
        albumentations.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albumentations.Compose(_transform)