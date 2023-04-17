import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import Satellite2Map_Data
from models import Generator, Discriminator
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
Gen_loss = []
Dis_loss = []

def train(netG, netD, train_dl, OptimizerG, OptimizerD, L1_Loss, BCE_Loss):
    loop = tqdm(train_dl)
    for idx, (x,y) in enumerate(loop):
        x = x.cuda()
        y = y.cuda()
        ############### Train discriminator ##############
        y_fake = netG(x)
        D_real = netD(x,y)
        D_real_loss = BCE_Loss(D_real, torch.ones_like(D_real))
        D_fake = netD(x,y_fake.detach())
        D_fake_loss = BCE_Loss(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss)/2

        netD.zero_grad()
        Dis_loss.append(D_loss.item())
        D_loss.backward()
        OptimizerD.step()

        ############## Train generator #############
        D_fake = netD(x, y_fake)
        G_fake_loss = BCE_Loss(D_fake, torch.ones_like(D_fake))
        L1 = L1_Loss(y_fake,y) * config.L1_LAMBDA
        G_loss = G_fake_loss + L1

        OptimizerG.zero_grad()
        Gen_loss.append(G_loss.item())
        G_loss.backward()
        OptimizerG.step()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

def main():
    netD = Discriminator(in_channels=3).cuda()
    netG = Generator(in_channels=3).cuda()
    OptimizerD = torch.optim.Adam(netD.parameters(),lr=config.LEARNING_RATE,betas=(config.BETA1,0.999))
    OptimizerG = torch.optim.Adam(netG.parameters(),lr=config.LEARNING_RATE,betas=(config.BETA1,0.999))
    BCE_Loss = nn.BCEWithLogitsLoss()
    L1_Loss = nn.L1Loss()
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,netG,OptimizerG,config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_DISC,netD,OptimizerD,config.LEARNING_RATE
        )
    train_dataset = Satellite2Map_Data(root=config.TRAIN_DIR)
    train_dl = DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS,pin_memory=True)
    val_dataset = Satellite2Map_Data(root=config.VAL_DIR)
    val_dl = DataLoader(val_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS,pin_memory=True)
    for epoch in range(config.NUM_EPOCHS):
        train(
            netG, netD, train_dl, OptimizerG, OptimizerD, L1_Loss, BCE_Loss
        )
        if config.SAVE_MODEL and epoch % 50 == 0:
            save_checkpoint(netG, OptimizerG, filename=config.CHECKPOINT_GEN)
            save_checkpoint(netD, OptimizerD, filename=config.CHECKPOINT_DISC)
        if epoch % 2 == 0:
            save_some_examples(netG, val_dl, epoch, folder="evaluation")
main()
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(Gen_loss, label="Generator")
plt.plot(Dis_loss, label="Discriminator")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig("figure1.png")