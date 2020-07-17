import torch
import torch.nn as nn
import numpy as np
import torchvision
import config, dataloader, ssim_loss
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import time
import visdom
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os



class discriminator(nn.Module):
    def __init__(self, alpha=0.2, dropout_value=0.0):
        super(discriminator, self).__init__()
        self.layers = nn.Sequential(
            # nn.LeakyReLU(alpha),
            nn.Conv2d(1, 64, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(alpha), nn.Dropout(dropout_value),
            nn.Conv2d(64, 128, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(alpha), nn.Dropout(dropout_value),
            nn.Conv2d(128, 256, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(alpha), nn.Dropout(dropout_value),
            nn.Conv2d(256, 512, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(alpha), nn.Dropout(dropout_value),
            nn.Conv2d(512, 1024, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(alpha), nn.Dropout(dropout_value),
            nn.AvgPool2d((4, 4), 1, 0),
        )
        self.fc = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()
        self.weight_init()

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
                nn.init.constant_(m.bias.data, 0)


class generater(nn.Module):
    def __init__(self, dropout_value=0.0):
        super(generater, self).__init__()
        # input = 4 * 4 * 1024
        # self.ic = input_channel
        # self.oc = output_channel
        self.feature_collect = feature_collect()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(), nn.Dropout(dropout_value),
            nn.ConvTranspose2d(512, 256, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(), nn.Dropout(dropout_value),
            nn.ConvTranspose2d(256, 128, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(), nn.Dropout(dropout_value),
            nn.ConvTranspose2d(128, 64, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(), nn.Dropout(dropout_value),
            nn.ConvTranspose2d(64, 1, (5, 5), (2, 2), 2, 1, bias=False),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Tanh(),
        )

        self.weight_init()

    def forward(self, x):
        x = self.feature_collect(x)
        x = self.layers(x)
        return x

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
                nn.init.constant_(m.bias.data, 0)


class feature_collect(nn.Module):
    def __init__(self):
        super(feature_collect, self).__init__()
        # input = 1*128*128 output = 1024*4*4
        res = torchvision.models.resnet18(pretrained=True)
        self.conv1 = nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.conv2 = nn.Conv2d(256, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        # print(res)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


if __name__ == '__main__':
    print(os.getcwd())
    writer = SummaryWriter('./log')

    vis = visdom.Visdom()

    lrelu_alpha = config.leakyrelu_alpha
    dropout_value = config.dropout_value

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_loader = DataLoader(dataset=dataloader.xDataSet(),
                              batch_size=config.batch_size,
                              num_workers=4,
                              drop_last=True)

    # Plot some training images
    # real_batch = next(iter(train_loader))
    # plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(
    #     np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    # plt.show()


    gen = generater(dropout_value=dropout_value).to(device)
    dis = discriminator(alpha=lrelu_alpha, dropout_value=dropout_value).to(device)
    g_optimizer = torch.optim.Adam(gen.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(dis.parameters(), lr=0.0002)
    loss = nn.BCELoss()
    sloss = ssim_loss.SSIM()

    # print(gen)
    # print(dis)

    g_loss = []
    d_loss = []

    if not os.path.exists('./model'):
        os.mkdir('./model')

    for epoch in range(config.epoch):
        gen_data = 0
        # nk_img = 0
        for batch_index, (NK_img, OK_img) in enumerate(train_loader):
            if batch_index is 0:
                writer.add_image('Input/Fake', vutils.make_grid(NK_img), global_step=epoch)
            # nk_img = ((NK_img.detach().cpu().numpy() + 1) * 127.5).astype(int)
            st = time.time()
            d_real_error = 0
            d_fake_error = 0
            d_acc_real = 0
            d_acc_fake = 0
            g_ssim = 0
            g_bce = 0
            g_d_gen_fake_acc = 0
            g_error = 0
            for d_ in range(config.d_steps):
                # Train D on real + fake
                dis.train()
                gen.eval()
                dis.zero_grad()

                # Train D on real
                d_real_data = OK_img.float().to(device)
                d_real_decision = dis(d_real_data)
                d_real_error = loss(d_real_decision, torch.ones([config.batch_size, 1]).to(device))
                d_real_error.backward()

                # Train D on fake
                d_fake_data = gen(NK_img.float().to(device))
                d_fake_decision = dis(d_fake_data)
                d_fake_error = loss(d_fake_decision, torch.zeros([config.batch_size, 1]).to(device))
                d_fake_error.backward()

                d_optimizer.step()

                d_loss.append(d_real_error.item()+d_fake_error.item())
                d_acc_real = d_real_decision.mean().item()
                d_acc_fake = d_fake_decision.mean().item()

                # vis.images(d_real_data, opts=dict(title='d-real'))
                # vis.images(d_fake_data, opts=dict(title='d-fake'))

            for g_ in range(config.g_steps):
                # Train G on D's response
                dis.eval()
                gen.train()
                gen.zero_grad()

                gen_data = gen(NK_img.float().to(device))
                d_g_fake_decision = dis(gen_data)
                g_error_bce = loss(d_g_fake_decision, torch.ones([config.batch_size, 1]).to(device))
                g_error_ssim = 1 - sloss(gen_data, OK_img.float().to(device))
                g_error = g_error_bce + g_error_ssim
                # g_error = (0.7*(1 - sloss(gen_data, item[1].float().to(device))) + 0.3*loss(d_g_fake_decision, torch.ones([config.batch_size, 1]).to(device)))
                g_error.backward()

                g_optimizer.step()

                g_loss.append(g_error_ssim.item())
                g_ssim = g_error_ssim.item()
                g_bce = g_error_bce.item()
                # g_bce = 0
                g_d_gen_fake_acc = d_g_fake_decision.mean().item()

            ed = time.time()
            print(
                'Train Epoch: {} [{:>2d}/{:>2d} ({:>2.0f}%)]\ttime:{:.2f}s\tg_Loss(bce/simm): {:.10f}/{:.10f}\td_Loss(real/fake): {:.10f}/{:.10f}\td_acc(real/fake): {:.7f}(1)/{:.7f}(0)\td_g_acc: {:.7f}(1)'.format(
                    epoch, batch_index + 1, len(train_loader),
                    100. * (batch_index + 1) / len(train_loader), ed - st, g_bce, g_ssim, d_real_error, d_fake_error, d_acc_real, d_acc_fake, g_d_gen_fake_acc))

        gen_images = ((gen_data.detach().cpu().numpy() + 1) * 127.5).astype(int)
        vis.images(gen_images)
        grid = vutils.make_grid(torch.from_numpy(gen_images))
        writer.add_image('Output/Generator', grid, global_step=epoch)
        writer.add_scalar('g_bce', g_bce, epoch)
        writer.add_scalar('g_simm', g_ssim, epoch)
        writer.add_scalar('d_real_error', d_real_error, epoch)
        writer.add_scalar('d_fake_error', d_fake_error, epoch)
        writer.add_scalar('d_acc_real', d_acc_real, epoch)
        writer.add_scalar('d_acc_fake', d_acc_fake, epoch)
        writer.add_scalar('g_d_gen_fake_acc', g_d_gen_fake_acc, epoch)
        writer.flush()
        if epoch % 5 == 0:
            torch.save(gen.state_dict(), './model/gen{}.ph'.format(epoch))
            torch.save(dis.state_dict(), './model/dis{}.ph'.format(epoch))

    writer.close()
    torch.save(gen.state_dict(), './model/gen.ph')
    torch.save(dis.state_dict(), './model/dis.ph')

