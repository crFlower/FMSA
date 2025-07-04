import torch
from PIL import Image
import torchvision
import os
from net import Network, vgg, FFCdecoder2, FFCdecoder2p

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load("checkpoint_180000.pkl", map_location=device)


decoder1 = FFCdecoder2p()
decoder2 = FFCdecoder2()
network = Network(vgg, decoder1, vgg, decoder2).cuda()
network.encoder.load_state_dict(checkpoint['encoder'])
network.decoder.load_state_dict(checkpoint['decoder'])
network.fft_transform1.load_state_dict(checkpoint['fft_transform1'])
network.fft_transform2.load_state_dict(checkpoint['fft_transform2'])
network.fft_transform3.load_state_dict(checkpoint['fft_transform3'])
network.ASA3_1.load_state_dict(checkpoint['ASA3_1'])
network.ASA4_1.load_state_dict(checkpoint['ASA4_1'])
network.ASA5_1.load_state_dict(checkpoint['ASA5_1'])
network.conv2d_1.load_state_dict(checkpoint['conv2d_1'])
network.conv2d_2.load_state_dict(checkpoint['conv2d_2'])
network.conv2d_3.load_state_dict(checkpoint['conv2d_3'])
network.transnet.load_state_dict(checkpoint['transmodule'])

# network.eval()

save_dir = "output"
os.makedirs(save_dir, exist_ok=True)

style_main_dir = "input/Style"
content_main_dir = "input/Content"
for stylefile in os.listdir(style_main_dir):
    stylepath=style_main_dir+"/"+stylefile
    ssimg = Image.open(stylepath).convert("RGB")  # 风格图
    for contentfile in os.listdir(content_main_dir):
        content_path = content_main_dir + "/" + contentfile
        ccimg = Image.open(content_path).convert("RGB")
        M = 512
        N = 512
        trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Resize((M, N))])


        cimg = trans(ccimg).unsqueeze(0).cuda()
        simg = trans(ssimg).unsqueeze(0).cuda()

        # 测试一：直接调用net
        _, _, cs, _, _, _ = network(cimg, simg)

        torchvision.utils.save_image(cs.detach().cpu(),
                                     save_dir + "/" + content_path.split('/')[-1][:-4] + "_stylized_" +
                                     stylefile.split('/')[-1])
