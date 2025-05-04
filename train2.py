import os, glob, cv2, numpy as np, math, torch, timm, warnings
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import albumentations as A
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationModel
from sklearn.model_selection import LeaveOneOut
from ema_pytorch import EMA
warnings.filterwarnings("ignore")

# ----------------- Config -----------------
DATA_DIR = "data"
RAW_H, RAW_W = 1162, 879
PATCH = 16
PAD_H, PAD_W = ((RAW_H-1)//PATCH+1)*PATCH, ((RAW_W-1)//PATCH+1)*PATCH
BATCH_SIZE   = 2          # 显存足够可 2
ACCUM_STEPS  = 1          # 若显存紧张可设 2
EPOCHS       = 80
WARMUP_EPOCH = 5
BASE_LR      = 5e-4       # decoder / projector
ENC_LR       = 1e-5       # ViT 微调
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS  = 4

# ----------------- File list --------------
AMP_DIR, PHA_DIR, MASK_DIR = [os.path.join(DATA_DIR,d) for d in ("amplitude","phase","masks")]
amp_files  = sorted(glob.glob(os.path.join(AMP_DIR,  "*.png")))
pha_files  = sorted(glob.glob(os.path.join(PHA_DIR,  "*.png")))
mask_files = sorted(glob.glob(os.path.join(MASK_DIR, "*.png")))
assert len(amp_files)==len(pha_files)==len(mask_files)==10, "请确认样本正好 10 对"

# --------------- Albumentations -----------
tf_train = A.Compose([
    A.HorizontalFlip(0.5), A.VerticalFlip(0.5),
    A.RandomRotate90(0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                       rotate_limit=15, border_mode=cv2.BORDER_CONSTANT,p=0.5),
    A.ColorJitter(0.2,0.2,0.2,0.1,p=0.4),
    A.OneOf([
        A.GaussNoise(var_limit=(10,50),p=0.3),
        A.ISONoise(p=0.3)
    ], p=0.3)
], additional_targets={'mask':'mask'})
tf_val = A.Compose([], additional_targets={'mask':'mask'})

def pad_hw(img):
    return cv2.copyMakeBorder(img,0,PAD_H-img.shape[0],0,PAD_W-img.shape[1],
                              cv2.BORDER_CONSTANT,0)

# --------------- Dataset ------------------
class ThzRGB(Dataset):
    def __init__(self,a,p,m,aug):
        self.a,self.p,self.m,self.aug=a,p,m,aug
    def __len__(self): return len(self.a)
    def __getitem__(self,idx):
        amp=cv2.cvtColor(cv2.imread(self.a[idx]),cv2.COLOR_BGR2RGB)
        pha=cv2.cvtColor(cv2.imread(self.p[idx]),cv2.COLOR_BGR2RGB)
        msk=cv2.imread(self.m[idx],cv2.IMREAD_GRAYSCALE)
        amp,pha,msk=[pad_hw(x) for x in (amp,pha,msk)]
        if self.aug:
            ret=self.aug(image=amp,mask=msk); amp,msk=ret["image"],ret["mask"]
            ret=self.aug(image=pha,mask=msk); pha=ret["image"]
        amp=torch.from_numpy(amp.transpose(2,0,1).astype("float32")/255.)
        pha=torch.from_numpy(pha.transpose(2,0,1).astype("float32")/255.)
        msk=torch.from_numpy((msk>0).astype("float32")[None])
        return amp,pha,msk

# --------------- ViT Encoder --------------
def build_vit():
    vit = timm.create_model(
        "vit_large_patch14_dinov2.lvd142m",
        pretrained=True,
        num_classes=0,
        in_chans=3
    )
    # 解冻最后 2 个 block
    for name, p in vit.named_parameters():
        p.requires_grad = name.startswith("blocks.22.") or name.startswith("blocks.23.")
    return vit


class ViTFeat(torch.nn.Module):
    def __init__(self,vit):
        super().__init__(); self.vit=vit; self.ps=16; self.C=vit.embed_dim
    def forward(self,x):
        f=self.vit.forward_features(x)[:,1:,:]
        h,w=x.shape[2]//self.ps,x.shape[3]//self.ps
        return [f.permute(0,2,1).reshape(-1,self.C,h,w)]

# --------------- Model --------------------
class DualViTUNet(SegmentationModel):
    def __init__(self,encA,encP,dec_ch=(768,512,256,128,64)):
        super().__init__()
        self.A,self.P=encA,encP
        enc_dim=encA.C*2
        self.decoder=UnetDecoder(
            encoder_channels=[enc_dim]*len(dec_ch),
            decoder_channels=dec_ch,n_blocks=len(dec_ch),
            use_batchnorm=True,center=True)
        self.head=torch.nn.Conv2d(dec_ch[-1],1,1)
    def forward(self,a,p):
        fA,fP=self.A(a)[0],self.P(p)[0]
        d=self.decoder(torch.cat([fA,fP],1))
        m=self.head(d)
        m=torch.nn.functional.interpolate(
            m,size=(PAD_H,PAD_W),mode="bilinear",align_corners=False)
        return m[...,:RAW_H,:RAW_W]

vitA,vitP=ViTFeat(build_vit()),ViTFeat(build_vit())
model=DualViTUNet(vitA.to(DEVICE),vitP.to(DEVICE)).to(DEVICE)

# optimizer with param groups
def param_groups(model):
    enc,dec=[],[]
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        (enc if "vit" in n else dec).append(p)
    return [{'params':dec,'lr':BASE_LR},
            {'params':enc,'lr':ENC_LR}]
opt=torch.optim.AdamW(param_groups(model),weight_decay=1e-4)
scaler=GradScaler()
ema=EMA(model,beta=0.9999,update_after_step=20)

# cosine schedule
def lr_lambda(epoch):
    if epoch<WARMUP_EPOCH: return (epoch+1)/WARMUP_EPOCH
    e=epoch-WARMUP_EPOCH
    return 0.5*(1+math.cos(math.pi*e/(EPOCHS-WARMUP_EPOCH)))
sch=torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

dice=smp.losses.DiceLoss("binary",from_logits=True)
bce =torch.nn.BCEWithLogitsLoss()

# -------------- Train loop ---------------
loo=LeaveOneOut()
for fold,(tr,va) in enumerate(loo.split(amp_files),1):
    print(f"\n=== Fold {fold}/10 ===")
    ds_tr=ThzRGB([amp_files[i] for i in tr],[pha_files[i] for i in tr],
                 [mask_files[i] for i in tr],tf_train)
    ds_va=ThzRGB([amp_files[i] for i in va],[pha_files[i] for i in va],
                 [mask_files[i] for i in va],tf_val)
    dl_tr=DataLoader(ds_tr,BATCH_SIZE,True,num_workers=NUM_WORKERS,pin_memory=True)
    dl_va=DataLoader(ds_va,1,False)
    best=9e9; early=0
    for ep in range(1,EPOCHS+1):
        # ---- train ----
        model.train(); epoch_loss=0
        for i,(a,p,m) in enumerate(dl_tr,1):
            a,p,m=[t.to(DEVICE) for t in (a,p,m)]
            with autocast():
                out=model(a,p); loss=(dice(out,m)+bce(out,m))/ACCUM_STEPS
            scaler.scale(loss).backward()
            if i%ACCUM_STEPS==0 or i==len(dl_tr):
                scaler.step(opt); scaler.update(); opt.zero_grad(); ema.update()
            epoch_loss+=loss.item()*ACCUM_STEPS
        # ---- val ----
        model.eval(); ema_model=ema.ema_model
        with torch.no_grad(), autocast():
            a=torch.from_numpy(np.zeros((1,3,RAW_H,RAW_W))).to(DEVICE) # placeholder
        val_l=[]
        for a,p,m in dl_va:
            a,p,m=[t.to(DEVICE) for t in (a,p,m)]
            with torch.no_grad(), autocast():
                out=ema_model(a,p)
                val_l.append((dice(out,m)+bce(out,m)).item())
        val_loss=np.mean(val_l)
        sch.step()
        print(f"Ep{ep:02d}/{EPOCHS}  LR={sch.get_last_lr()[0]:.2e}  "
              f"train={epoch_loss/len(dl_tr):.4f}  val={val_loss:.4f}")
        # ---- save ----
        if val_loss<best:
            best,val_best=val_loss,ep; early=0
            torch.save(ema_model.state_dict(),f"best_fold{fold}.pth")
            print("  ** best **")
        else:
            early+=1
            if early>=15: print("  Early stop"); break
    print(f"Fold{fold} best val {best:.4f} @ Ep{val_best}")

print("\n>>> 全部 10‑fold 训练完成，权重文件 best_fold*.pth")
