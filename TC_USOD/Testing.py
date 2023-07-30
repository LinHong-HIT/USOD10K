import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import get_loader
import transforms as trans
from torchvision import transforms
import time
from Models.USOD_Net import ImageDepthNet
from torch.utils import data
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image


def test_net(args):

    cudnn.benchmark = True

    net = ImageDepthNet(args)
    net.cuda()
    net.eval()

    # load model (multi-gpu)
    model_path = args.save_model_dir + 'TC_USOD.pth'
    print(model_path)
    state_dict = torch.load(model_path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)

    print('Model loaded from {}'.format(model_path))

    test_paths = args.test_paths.split('+')

    for test_dir_img in test_paths:

        test_dataset = get_loader(test_dir_img, args.data_root, args.img_size, mode='test')

        test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)
        print('''
                   Starting testing:
                       dataset: {}
                       Testing size: {}
                   '''.format(test_dir_img.split('/')[0], len(test_loader.dataset)))

        time_list = []
        for i, data_batch in enumerate(test_loader):
            images, depths, image_w, image_h, image_path = data_batch
            images, depths = Variable(images.cuda()), Variable(depths.cuda())

            starts = time.time()
            outputs_saliency= net(images, depths)
            ends = time.time()
            time_use = ends - starts
            time_list.append(time_use)

            d1, d2, d3, d4, d5, db, ud2, ud3, ud4, ud5, udb = outputs_saliency
            image_w, image_h = int(image_w[0]), int(image_h[0])
            transform = trans.Compose([
                transforms.ToPILImage(),
                trans.Scale((image_w, image_h))
            ])
            output_s = d1.data.cpu().squeeze(0)

            output_s = transform(output_s)

            dataset = test_dir_img.split('/')[0]
            filename = image_path[0].split('/')[-1].split('.')[0]

            # save saliency maps
            save_test_path = args.save_test_path_root + dataset + '/USOD10K/'
            if not os.path.exists(save_test_path):
                os.makedirs(save_test_path)
            output_s.save(os.path.join(save_test_path, filename + '.png'))
        print('dataset:{}, cost:{}'.format(test_dir_img.split('/')[0], np.mean(time_list)*1000))






