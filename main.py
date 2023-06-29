import itertools
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model import Generator_Res, Discriminator_Patch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator_Res().to(device)
generator2 = Generator_Res().to(device)
discriminator = Discriminator_Patch().to(device)
discriminator2 = Discriminator_Patch().to(device)

data = torch.load('data.pth')
generator.load_state_dict(data['generator'])
generator2.load_state_dict(data['generator2'])
discriminator.load_state_dict(data['discriminator'])
discriminator2.load_state_dict(data['discriminator2'])

criterionMSE = nn.MSELoss()
criterionMAE = nn.L1Loss()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(3)], [0.5 for _ in range(3)])
])

lr = 2e-4
batch_size = 1
num_epochs = 10
#epochs = 0
epochs = data['epochs']


writer1 = SummaryWriter(f'runs/anime')
writer2 = SummaryWriter(f'runs/human')
writerf1 = SummaryWriter('runs/human2anime')
writerf2 = SummaryWriter('runs/anime2human')

anime = datasets.ImageFolder("dataset/anime", transform=transform)
human = datasets.ImageFolder("dataset/human", transform=transform)

loader_anime = DataLoader(anime, batch_size=batch_size, shuffle=True)
loader_human = DataLoader(human, batch_size=batch_size, shuffle=True)

gen_Optimizer = torch.optim.Adam(itertools.chain(generator.parameters(), generator2.parameters()),
                                 lr=lr, betas=(0.5, 0.999))

disc_Optimizer = torch.optim.Adam(itertools.chain(discriminator.parameters(), discriminator2.parameters()),
                                  lr=lr, betas=(0.5, 0.999))

gen_Optimizer.load_state_dict(data['gen_Optimizer'])
disc_Optimizer.load_state_dict(data['disc_Optimizer'])
step = 0
generator2.train()
generator.train()
discriminator.train()
discriminator2.train()

# only for learning rate editing purpose no need to use if not necessary
#gen_Optimizer.param_groups[0]['lr'] = lr
#disc_Optimizer.param_groups[0]['lr'] = lr
for epoch in range(num_epochs):
    if epoch+epochs >= 100:
        # learning rate decay after 100 epoch
        gen_Optimizer.param_groups[0]['lr'] *= 0.954991
        disc_Optimizer.param_groups[0]['lr'] *= 0.954991
    for batch_idx, (real_anime, real_human) in enumerate(zip(loader_anime, loader_human)):
        real_anime = real_anime[0].to(device)
        real_human = real_human[0].to(device)

        gen_Optimizer.zero_grad()
        a_fake = generator2(real_human)
        disc_pred = discriminator2(a_fake)

        label1 = torch.ones(disc_pred.size()).to(device)

        b_generator_loss = criterionMSE(disc_pred, label1)

        b_recon = generator(a_fake)
        b_cycle_loss = criterionMAE(b_recon, real_human) * 10

        b_idt = generator2(real_anime)
        b_idt_losses = criterionMAE(b_idt, real_human) * 10 * 0.5

        b_total_loss = b_generator_loss + b_cycle_loss + b_idt_losses
        b_total_loss.backward(retain_graph=True)
        gen_Optimizer.step()

        gen_Optimizer.zero_grad()
        h_fake = generator(real_anime)
        discb_pred = discriminator(h_fake)
        label1 = torch.ones(discb_pred.size()).to(device)
        a_generator_loss = criterionMSE(discb_pred, label1)

        a_recon = generator2(h_fake)
        a_cycle_loss = criterionMAE(a_recon, real_anime) * 10

        a_idt = generator(real_human)
        a_idt_loss = criterionMAE(a_idt, real_anime) * 10 * 0.5

        a_total_loss = a_generator_loss + a_cycle_loss + a_idt_loss
        a_total_loss.backward(retain_graph=True)
        gen_Optimizer.step()

        disc_Optimizer.zero_grad()
        b_real_dis = discriminator2(real_anime)
        dlabel_1 = torch.ones(b_real_dis.size()).to(device)
        b_dis_real_loss = criterionMSE(b_real_dis, dlabel_1)

        b_fake_dis = discriminator2(a_fake)
        dlabel_0 = torch.zeros(b_fake_dis.size()).to(device)
        b_dis_fake_loss = criterionMSE(b_fake_dis, dlabel_0)

        b_dis_loss = (b_dis_real_loss + b_dis_fake_loss) * 0.5
        b_dis_loss.backward()
        disc_Optimizer.step()

        disc_Optimizer.zero_grad()
        a_real_dis = discriminator(real_human)
        dlabel_1 = torch.ones(a_real_dis.size()).to(device)

        a_dis_real_loss = criterionMSE(a_real_dis, dlabel_1)

        a_fakedis = discriminator(h_fake)
        dlabel_0 = torch.zeros(a_fakedis.size()).to(device)
        a_dis_fake_loss = criterionMSE(a_fakedis, dlabel_0)

        a_dis_loss = (a_dis_real_loss + a_dis_fake_loss) * 0.5
        a_dis_loss.backward()
        disc_Optimizer.step()

        if batch_idx % 200 == 0:
            print(f"Epoch [{epoch + 1 + epochs}/{num_epochs+epochs}] Batch {batch_idx}/{len(loader_anime)} \
                        Loss DiscA: {a_dis_loss:.4f}, Loss DiscB: {b_dis_loss:.4f} \n"
                  f"loss_GenA: {a_total_loss:.4f}, loss_GenB: {b_total_loss:.4f}"

                  )
            with torch.no_grad():
                img_grid_real = torchvision.utils.make_grid(
                    real_anime, normalize=True
                )   
                img_grid_fake = torchvision.utils.make_grid(
                    a_fake, normalize=True
                )
                img_grid_real1 = torchvision.utils.make_grid(
                    real_human, normalize=True
                )
                img_grid_fake1 = torchvision.utils.make_grid(
                    h_fake, normalize=True
                )
                writer1.add_image("Anime", img_grid_real, global_step=step)
                writerf1.add_image("Human2Anime", img_grid_fake, global_step=step)
                writer2.add_image("Human", img_grid_real1, global_step=step)
                writerf2.add_image("Anime2Human", img_grid_fake1, global_step=step)
            step += 1
data = {
    'epochs': epochs + num_epochs,
    'generator': generator.state_dict(),
    'generator2': generator2.state_dict(),
    'discriminator': discriminator.state_dict(),
    'discriminator2': discriminator2.state_dict(),
    'disc_Optimizer': disc_Optimizer.state_dict(),
    'gen_Optimizer': gen_Optimizer.state_dict()
}
torch.save(data, 'data.pth')
