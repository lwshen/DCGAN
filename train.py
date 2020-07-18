import config, dataloader, ssim_loss, model
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import visdom
import os
import time

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

    gen = model.generater(dropout_value=dropout_value).to(device)
    dis = model.discriminator(alpha=lrelu_alpha, dropout_value=dropout_value).to(device)
    g_optimizer = torch.optim.Adam(gen.parameters(), lr=config.lr)
    d_optimizer = torch.optim.Adam(dis.parameters(), lr=config.lr)
    loss = nn.BCELoss()
    sloss = ssim_loss.SSIM()
    l1loss = nn.L1Loss()

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
                g_error_l1 = l1loss(gen_data, OK_img.float().to(device))
                g_error = g_error_bce + g_error_ssim * 10 + g_error_l1 * 10
                # g_error = (0.7*(1 - sloss(gen_data, item[1].float().to(device))) + 0.3*loss(d_g_fake_decision, torch.ones([config.batch_size, 1]).to(device)))
                g_error.backward()

                g_optimizer.step()

                g_loss.append(g_error.item())
                g_ssim = g_error_ssim.item()
                g_bce = g_error_bce.item()
                g_l1 = g_error_l1.item()
                # g_bce = 0
                g_all = g_error.item()
                g_d_gen_fake_acc = d_g_fake_decision.mean().item()

            ed = time.time()
            print(
                'Train Epoch: {} [{:>2d}/{:>2d} ({:>2.0f}%)]\ttime:{:.2f}s\tg_Loss(bce/simm/l1/all): {:.10f}/{:.10f}/{:.10f}/{:.10f}\td_Loss(real/fake): {:.10f}/{:.10f}\td_acc(real/fake): {:.7f}(1)/{:.7f}(0)\td_g_acc: {:.7f}(1)'.format(
                    epoch, batch_index + 1, len(train_loader),
                    100. * (batch_index + 1) / len(train_loader), ed - st, g_bce, g_ssim, g_l1, g_all, d_real_error, d_fake_error, d_acc_real, d_acc_fake, g_d_gen_fake_acc))

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

