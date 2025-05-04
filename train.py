import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm
import warnings # Import the warnings module

warnings.filterwarnings('ignore', message='.*iCCP: known incorrect sRGB profile.*')

# ------------------------------
# 1. 配置
# ------------------------------
DATA_DIR    = 'data'                   # 根目录（包含 amplitude、phase、masks 三个子目录）
AMP_DIR     = os.path.join(DATA_DIR, 'amplitude')
PHA_DIR     = os.path.join(DATA_DIR, 'phase')
MASK_DIR    = os.path.join(DATA_DIR, 'masks')

TARGET_SIZE = (1162, 879)   # 重采样尺寸
BATCH_SIZE  = 2
EPOCHS      = 30
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------------------
# 2. 构建文件列表（确保按相同顺序）
# ------------------------------
amp_files  = sorted(glob.glob(os.path.join(AMP_DIR,  '*.png')))
pha_files  = sorted(glob.glob(os.path.join(PHA_DIR,  '*.png')))
mask_files = sorted(glob.glob(os.path.join(MASK_DIR, '*.png')))
assert len(amp_files)==len(pha_files)==len(mask_files), "振幅/相位/掩码 数量不匹配"

# ------------------------------
# 3. 数据增强 & 变换
# ------------------------------
train_transform = A.Compose([
    A.RandomResizedCrop(height=TARGET_SIZE[0], width=TARGET_SIZE[1],
                        scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.7),

    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                       rotate_limit=25, border_mode=cv2.BORDER_CONSTANT, p=0.5),

    A.ElasticTransform(alpha=20, sigma=50, alpha_affine=10, p=0.2),

    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0)),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True)
    ], p=0.3),

    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.2),

    A.MotionBlur(blur_limit=3, p=0.1),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
], additional_targets={'mask':'mask'})


val_transform = A.Compose([
    A.Resize(*TARGET_SIZE)
], additional_targets={'mask':'mask'})

# ------------------------------
# 4. 自定义 Dataset
# ------------------------------
class ThzDataset(Dataset):
    def __init__(self, amp_list, pha_list, mask_list, transform=None):
        self.amp_list  = amp_list
        self.pha_list  = pha_list
        self.mask_list = mask_list
        self.transform = transform

    def __len__(self):
        return len(self.amp_list)

    def __getitem__(self, idx):
        amp = cv2.imread(self.amp_list[idx],  cv2.IMREAD_GRAYSCALE)
        pha = cv2.imread(self.pha_list[idx],  cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(self.mask_list[idx], cv2.IMREAD_GRAYSCALE)
        img = np.stack([amp, pha], axis=-1)  # H×W×2

        if self.transform:
            aug = self.transform(image=img, mask=msk)
            img, msk = aug['image'], aug['mask']

        # 归一化 & 顺序调整
        img = img.astype('float32') / 255.0
        img = img.transpose(2,0,1)            # 2×H×W

        # 掩码二值化 & 添加通道
        msk = (msk > 0).astype('float32')
        msk = np.expand_dims(msk, 0)          # 1×H×W

        return torch.from_numpy(img), torch.from_numpy(msk)

# ------------------------------
# 5. 构建模型、损失、优化器、调度器
# ------------------------------
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=2,
    classes=1,
    activation=None
)
model.to(DEVICE)

# Define loss functions separately
dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True) 
bce_loss  = torch.nn.BCEWithLogitsLoss() 

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# ------------------------------
# 6. LOOCV 训练循环
# ------------------------------
loo = LeaveOneOut()
for fold, (train_idx, val_idx) in enumerate(loo.split(amp_files), start=1):
    print(f'\n=== Fold {fold}/{len(amp_files)} ===')
    # 划分数据
    train_amp = [amp_files[i] for i in train_idx]
    train_pha = [pha_files[i] for i in train_idx]
    train_msk = [mask_files[i] for i in train_idx]
    val_amp   = [amp_files[i] for i in val_idx]
    val_pha   = [pha_files[i] for i in val_idx]
    val_msk   = [mask_files[i] for i in val_idx]

    train_ds = ThzDataset(train_amp, train_pha, train_msk, transform=train_transform)
    val_ds   = ThzDataset(val_amp,   val_pha,   val_msk,   transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)

    best_val_loss = float('inf')
    train_hist, val_hist = [], []   # 记录历史
    for epoch in range(1, EPOCHS+1):
        # 训练
        model.train()
        t_losses = []
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds       = model(imgs)
            # Calculate and combine loss values
            loss1       = dice_loss(preds, masks)
            loss2       = bce_loss(preds, masks)
            loss        = loss1 + loss2 # Add the loss results
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_losses.append(loss.item())
        train_loss = np.mean(t_losses)
        train_hist.append(train_loss)
        val_hist.append(val_loss)


        # 验证
        model.eval()
        v_losses = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds       = model(imgs)
                 # Calculate and combine loss values
                loss1       = dice_loss(preds, masks)
                loss2       = bce_loss(preds, masks)
                loss        = loss1 + loss2 # Add the loss results
                v_losses.append(loss.item())
        val_loss = np.mean(v_losses)
        scheduler.step(val_loss)

        print(f'Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}')

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'model_fold{fold}.pth')
            print('  >>> Saved best model')

    np.save(f'history_fold{fold}.npy', np.vstack([train_hist, val_hist]))

print('\n全部 Fold 训练完成，权重保存在 model_fold1.pth … model_fold5.pth')
