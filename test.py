import torch
from torch.utils.data import DataLoader
import model, dataloader, config
import torchvision.utils as vutils
import visdom
import time
import os

if __name__ == '__main__':
    # vis = visdom.Visdom()
    iter = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    test_loader = DataLoader(dataset=dataloader.tDataSet(),
                              batch_size=config.test_batch_size,
                              num_workers=4,
                              drop_last=True)

    gen = model.generater(dropout_value=0.0)
    gen.load_state_dict(torch.load(config.ModelPath, map_location=device))
    gen.to(device)
    gen.eval()
    start_time = time.time()
    for batch_index, NK_img in enumerate(test_loader):
        st = time.time()
        output = gen(NK_img.float().to(device))
        ed = time.time()
        print(
            'Test: [{:>2d}/{:>2d} ({:>2.0f}%)]\ttime:{:.2f}s'.format(
                batch_index + 1, len(test_loader),
                100. * (batch_index + 1) / len(test_loader), ed - st))
        gen_images = ((output.detach().cpu().numpy() + 1) * 127.5).astype(int)
        test_images = ((NK_img.detach().cpu().numpy() + 1) * 127.5).astype(int)
        if config.save_single_image:
            if not os.path.exists('./output'):
                os.mkdir('./output')
            if not os.path.exists('./output/gen'):
                os.mkdir('./output/gen')
            if not os.path.exists('./output/test'):
                os.mkdir('./output/test')
            for i in range(len(gen_images)):
                vutils.save_image(torch.from_numpy(gen_images[i]/255.0), './output/gen/%05d.jpg' % (iter), padding=0)
                vutils.save_image(torch.from_numpy(test_images[i]/255.0), './output/test/%05d.jpg' % (iter), padding=0)
                iter += 1
        if config.save_grid_image:
            if not os.path.exists('./output'):
                os.mkdir('./output')
            if not os.path.exists('./output/grid'):
                os.mkdir('./output/grid')
            gen_grid = vutils.make_grid(torch.from_numpy(gen_images))
            test_grid = vutils.make_grid(torch.from_numpy(test_images))
            vutils.save_image(gen_grid/255.0, './output/grid/{}gen.jpg'.format(batch_index), padding=0)
            vutils.save_image(test_grid/255.0, './output/grid/{}test.jpg'.format(batch_index), padding=0)
        # vis.images(test_images)
        # vis.images(gen_images)

    end_time = time.time()
    print('All time use: {.2f}'.format(end_time - start_time))