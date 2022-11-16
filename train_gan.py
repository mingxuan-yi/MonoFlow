import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.utils as vutils
import numpy as np
from source.models import weights_init
from source.data import get_dataset, DataTransform
from source.nets import DCG_mnist, DCG_cifar10, Discriminator, Discriminator_mnist 
import argparse
import os
import matplotlib.pyplot as plt
torch.manual_seed(321)
torch.cuda.manual_seed_all(321)
#torch.use_deterministic_algorithms(True)

criterion = nn.BCEWithLogitsLoss()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], default='mnist')
    parser.add_argument('--dataroot', type=str, default='../data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--method', type=str, choices=['ns', 'vanilla', 'logit'], default='vanilla')
    parser.add_argument('--C', type=float, default=5.0)
    parser.add_argument('--saveroot', type=str, default='./saved')
    

    parser.add_argument('--d_lr', type=float, default=0.0002)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--dim_z', type=float, default=100)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--ngpu', type=int, default=1)


    return parser.parse_args()



    
def train(x_real, netG, netD, D_optimizer, G_optimizer, device, method='kl', niter=1):
    bs = x_real.shape[0]
    noise_z = torch.randn(bs, args.dim_z, 1, 1, device=device)
    x_fake = netG(noise_z)
    label1 = torch.ones(bs, 1).to(device)
    label0 = torch.zeros(bs, 1).to(device)
    #t = torch.zeros(bs).to(device)
    #=======================Train the discriminator=======================#
    for i in range(niter):
        D_optimizer.zero_grad()
        
        D_real_loss = criterion(netD(x_real), label1)
        D_fake_loss = criterion(netD(x_fake.detach()), label0)
       

        # gradient backprop & optimize ONLY D's parameters
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D_optimizer.step()

    #=======================Train the generator=======================#
    G_optimizer.zero_grad()
    d = netD(x_fake) #log ratio
    if method == 'ns':
        G_loss = -torch.mean(torch.log(torch.sigmoid(d))) #ns gan
    elif method == 'vanilla':
        G_loss = torch.mean(torch.log(1.0 - torch.sigmoid(d+args.C))) #original gan
    elif method == 'logit':
        G_loss = -torch.mean(d)
    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return  D_loss.data.item(), d.mean().item() # return kl 


def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)
    data_process = DataTransform(None)
    # Prepare the dataset
    data = get_dataset(args.dataroot, args.dataset)
    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True)
    
    # Define the d(x)
    if args.dataset == 'mnist':
        netD = Discriminator_mnist(args, ndf=32, nc=1).to(device)
        netG = DCG_mnist(nz=args.dim_z, ngf=32, nc=1).to(device)
    elif args.dataset == 'cifar10':
        netD = Discriminator(args, ndf=64, nc=3).to(device)
        netG = DCG_cifar10(nz=args.dim_z, ngf=64, nc=3).to(device)

    if (device.type == 'cuda') and (args.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(args.ngpu)))
        netG = nn.DataParallel(netG, list(range(args.ngpu)))
    
    netD.apply(weights_init)
    netG.apply(weights_init)
    D_optimizer = optim.Adam(netD.parameters(), lr = args.d_lr, betas=(args.beta1, 0.999))
    G_optimizer = optim.Adam(netG.parameters(), lr = args.g_lr, betas=(args.beta1, 0.999))

    img_visual_path = os.path.join(args.saveroot, "visual")
    model_path = os.path.join(args.saveroot, "saved_models")

    if not(os.path.isdir(args.saveroot)):
        os.mkdir(args.saveroot)
        os.mkdir(img_visual_path)
        os.mkdir(model_path)

    dpath = os.path.join(model_path, f"gan_D_latest.pth")
    gpath = os.path.join(model_path, f"gan_G_latest.pth")
    if os.path.isfile(dpath) and os.path.isfile(gpath):
        netD.load_state_dict(torch.load(dpath))
        netG.load_state_dict(torch.load(gpath))
        print('Loaded previous models')

    for epoch in range(0, args.num_epochs):
        
        D_losses, G_losses = [], []
        for batch_idx, (x, _) in enumerate(data_loader):
            x_real = data_process.forward_transform(x).to(device)

            d_loss, g_loss = train(x_real, netG, netD, D_optimizer, G_optimizer, device, method=args.method)
            D_losses.append(d_loss)
            G_losses.append(g_loss)
        print('[%d/%d]: loss_d: %.4f, loss_g: %.4f' % (
                    (epoch), args.num_epochs, torch.mean(torch.FloatTensor(D_losses)), g_loss))
        with torch.no_grad():
            fake_plot = netG(fixed_noise).detach().cpu()
            img = vutils.make_grid(fake_plot[:36], nrow=6, padding=1, normalize=True)

        filepath_fig = os.path.join(img_visual_path, f"gan-{epoch}.png")
        #filepath_D = os.path.join(model_path, f"gan_D-{epoch}.pth")
        #filepath_G = os.path.join(model_path, f"gan_G-{epoch}.pth")
        #torch.save(netD.state_dict(), filepath_D)
        #torch.save(netG.state_dict(), filepath_G)

        plt.figure(figsize=(5,5))
        
        plt.axis("off")
        #plt.title("Fake Images")
        plt.imshow(np.transpose(img,(1,2,0)))
        plt.savefig(filepath_fig, bbox_inches='tight', pad_inches=0)
        plt.close() 
    torch.save(netD.state_dict(), dpath)
    torch.save(netG.state_dict(), gpath)


if __name__ == '__main__':
    args = get_args()
    main(args)