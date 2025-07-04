import torch
import torch.nn as nn
import torch.nn.functional as F
from FFC import FFC_BN_ACT

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1   3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1  10--9
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1   17--15
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used   30--27
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1  43--39
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

# compute channel-wise means and variances of features
def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4, 'The shape of feature needs to be a tuple with length 4.'
    B, C = size[:2]
    feat_mean = feat.reshape(B, C, -1).mean(dim=2).reshape(B, C, 1, 1)
    feat_std = (feat.reshape(B, C, -1).var(dim=2) + eps).sqrt().reshape(B, C, 1, 1)
    return feat_mean, feat_std


# normalize features
def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

class fftTransform(nn.Module):
    def __init__(self, in_planes, H, W):
        super(fftTransform, self).__init__()
        self.in_planes = in_planes
        self.H = H
        self.W = W
        self.complex_weight_c = nn.Parameter(torch.randn(in_planes, H, W, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_s = nn.Parameter(torch.randn(in_planes, H, W, 2, dtype=torch.float32) * 0.02)



    def forward(self, FFTc, FFTs):
        b, c, h, w = FFTc.shape

        weight_c = self.complex_weight_c.permute(0, 3, 1, 2)
        weight_s = self.complex_weight_s.permute(0, 3, 1, 2)


        weight_c_resized = F.interpolate(weight_c, size=(h, w), mode='bilinear', align_corners=False)
        weight_s_resized = F.interpolate(weight_s, size=(h, w), mode='bilinear', align_corners=False)


        weight_c_complex = torch.complex(weight_c_resized[:, 0], weight_c_resized[:, 1])  # [C, H, W]
        weight_s_complex = torch.complex(weight_s_resized[:, 0], weight_s_resized[:, 1])  # [C, H, W]

        weight_c_complex = weight_c_complex.unsqueeze(0)
        weight_s_complex = weight_s_complex.unsqueeze(0)

        FFTc_out = FFTc * weight_c_complex
        FFTs_out = FFTs * weight_s_complex

        return FFTc_out, FFTs_out

class ASA(nn.Module):

    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        super(ASA, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None, content_mask=None, style_mask=None):
        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        if style_mask is not None:
            style_mask = nn.functional.interpolate(
                style_mask, size=(h_g, w_g), mode='nearest').view(b, 1, w_g * h_g).contiguous()
        else:
            style_mask = torch.ones(b, 1, w_g * h_g, device=style.device)
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_mask = style_mask[:, :, index]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        if content_mask is not None:
            content_mask = nn.functional.interpolate(
                content_mask, size=(h, w), mode='nearest').view(b, 1, w * h).permute(0, 2, 1).contiguous()
        else:
            content_mask = torch.ones(b, w * h, 1, device=content.device)
        S = torch.bmm(F, G)
        style_mask = 1. - style_mask
        attn_mask = torch.bmm(content_mask, style_mask)
        S = S.masked_fill(attn_mask.bool(), -1e15)
        # S: b, n_c, n_s
        S = self.sm(S)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return std * mean_variance_norm(content) + mean


class FFCdecoder2p(nn.Module):
    def __init__(self):
        super(FFCdecoder2p, self).__init__()
        self.conv1 = FFC_BN_ACT(512, 256)
        self.upsample1=nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = FFC_BN_ACT(256, 256)
        self.conv3 = FFC_BN_ACT(256, 256)
        self.conv4 = FFC_BN_ACT(256, 256)

    def forward(self,x):
        out=self.conv1(x)
        out=self.upsample1(out)
        out=self.conv2(out)
        out=self.conv3(out)
        out=self.conv4(out)

        return out

class FFCdecoder2(nn.Module):
    def __init__(self):
        super(FFCdecoder2, self).__init__()

        self.conv5=FFC_BN_ACT(256,128)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6 = FFC_BN_ACT(128, 128)
        self.conv7 = FFC_BN_ACT(128,64)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8 = FFC_BN_ACT(64, 64)
        self.conv=nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, 3, 1, 0)
        )
    def forward(self,x):

        out=self.conv5(x)
        out = self.upsample2(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.upsample3(out)
        out = self.conv8(out)
        out=self.conv(out)

        return out


class Network(nn.Module):
    def __init__(self,Encoder,Decoder,lossNet,transnet=None):
        super(Network, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss=nn.L1Loss()
        self.encoder = Encoder
        self.decoder = Decoder
        self.transnet=transnet
        self.sigmoid=nn.Sigmoid()
        self.relu=nn.ReLU()
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.fft_transform1 = fftTransform(256, 56, 56)
        self.fft_transform2 = fftTransform(512, 28, 28)
        self.fft_transform3 = fftTransform(512, 14, 14)
        self.ASA3_1 = ASA(in_planes=256, key_planes=256 + 128 + 64)
        self.ASA4_1 = ASA(in_planes=512, key_planes=512 + 256 + 128 + 64)
        self.ASA5_1 = ASA(in_planes=512, key_planes=512 + 256 + 128 + 64 + 512)
        self.conv2d_1 = nn.Conv2d(512, 256, (1, 1))
        self.conv2d_2 = nn.Conv2d(1024, 512, (1, 1))
        self.conv2d_3 = nn.Conv2d(1024, 512, (1, 1))

        self.enclayer = list(self.encoder.children())
        self.enc1 = nn.Sequential(*self.enclayer[:4])
        self.enc2 = nn.Sequential(*self.enclayer[4:11])
        self.enc3 = nn.Sequential(*self.enclayer[11:18])
        self.enc4 = nn.Sequential(*self.enclayer[18:31])
        self.enc5 = nn.Sequential(*self.enclayer[31:44])
        image_encoder_layers = [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]
        for name in ['enc1', 'enc2', 'enc3', 'enc4', 'enc5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False




        # features of intermediate layers
        lossNet_layers = list(lossNet.children())
        self.feat_1 = nn.Sequential(*lossNet_layers[:4])  # input -> relu1_1
        self.feat_2 = nn.Sequential(*lossNet_layers[4:11])  # relu1_1 -> relu2_1
        self.feat_3 = nn.Sequential(*lossNet_layers[11:18])  # relu2_1 -> relu3_1
        self.feat_4 = nn.Sequential(*lossNet_layers[18:31])  # relu3_1 -> relu4_1
        self.feat_5 = nn.Sequential(*lossNet_layers[31:44])  # relu3_1 -> relu4_1

        for name in ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
        self.sobel_x = torch.tensor([[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).expand(
            (1, 3, 3, 3)).to(
            torch.device("cuda"))

        self.sobel_y = torch.tensor([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).expand(
            (1, 3, 3, 3)).to(
            torch.device("cuda"))
        self.lap=torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]],dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).expand(
            (1, 3, 3, 3)).to(
            torch.device("cuda"))
        self.pad=nn.ReflectionPad2d((1,1,1,1))

    # get intermediate features
    def get_interal_feature(self, input):
        result = []
        for i in range(5):
            input = getattr(self, 'feat_{:d}'.format(i + 1))(input)
            result.append(input)

        return result

    def get_key(self, feats, last_layer_idx):
        results = []
        _, _, h, w = feats[last_layer_idx].shape
        for i in range(last_layer_idx):
            results.append(mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
        results.append(mean_variance_norm(feats[last_layer_idx]))
        return torch.cat(results, dim=1)

    #风格内容融合
    def transmodule(self,FFC,FFS):
        AFFC=torch.sqrt(FFC.real ** 2 + FFC.imag ** 2 + 1e-16)
        AFFS=torch.sqrt(FFS.real ** 2 + FFS.imag ** 2 + 1e-16)
        FFCScos=FFC.real/AFFC
        FFCSsin=FFC.imag/AFFC
        FFCSreal=FFCScos*AFFS
        FFCSimag=FFCSsin*AFFS
        FFCS=torch.complex(FFCSreal,FFCSimag)
        return FFCS


    #内容损失
    def calc_content_loss(self,scfftr1,cfftr1,norm=False):
        scfftr=scfftr1
        cfftr=cfftr1
        scfft=torch.fft.fft2(scfftr,dim=(-2,-1))
        cfft=torch.fft.fft2(cfftr,dim=(-2,-1))
        B, N, H, W = cfft.shape
        distance=torch.sqrt((cfft.real-scfft.real)**2+(cfft.imag-scfft.imag)**2+1e-16)
        minus=torch.zeros((distance.shape)).cuda()
        return self.mse_loss(distance,minus)/(H*W)

    def calc_style_loss(self,scimg,simg):

        B, N, H, W = scimg.shape
        scfft=torch.fft.fft2(scimg,dim=(-2,-1))
        sfft=torch.fft.fft2(simg,dim=(-2,-1))
        sfftenergy=sfft.real**2+sfft.imag**2
        scfftenergy=scfft.real**2+scfft.imag**2
        # sfftenergy[torch.isnan(sfftenergy)]=0.0
        # scfftenergy[torch.isnan(scfftenergy)]=0.0

        return self.l1_loss(sfftenergy,scfftenergy)/(H*W)


    def calc_sum_loss(self, oimg, csimg):
        foimg = torch.fft.fft2(oimg)
        fcsimg = torch.fft.fft2(csimg)
        return self.mse_loss(foimg.real,fcsimg.real)+self.mse_loss(foimg.imag,fcsimg.imag)

    def calc_content_loss_variance(self, sc, c):
        sc_var = mean_variance_norm(sc)
        c_var = mean_variance_norm(c)
        return self.mse_loss(sc_var, c_var)

    def calc_style_loss_gloal_mean_std(self, sc, s):
        sc_mean, sc_std = calc_mean_std(sc)
        s_mean, s_std = calc_mean_std(s)
        return self.mse_loss(sc_mean, s_mean) + self.mse_loss(sc_std, s_std)

    def calc_style_loss_local_mean_std(self, sc_feats, s_feats, c_feats):
        loss_local = 0
        for i in range(1, 5):
            c_key = self.get_key(c_feats, i)
            s_key = self.get_key(s_feats, i)
            s_value = s_feats[i]
            b, _, h_s, w_s = s_key.size()
            s_key = s_key.view(b, -1, h_s * w_s).contiguous()
            if h_s * w_s > 256 * 256:
                torch.manual_seed(6666)
                index = torch.randperm(h_s * w_s).to('cuda')[:256 * 256]
                s_key = s_key[:, :, index]
                style_flat = s_value.view(b, -1, h_s * w_s)[:, :, index].transpose(1, 2).contiguous()
            else:
                style_flat = s_value.view(b, -1, h_s * w_s).transpose(1, 2).contiguous()
            b, _, h_c, w_c = c_key.size()
            c_key = c_key.view(b, -1, h_c * w_c).permute(0, 2, 1).contiguous()
            attn = torch.bmm(c_key, s_key)
            # S: b, n_c, n_s
            attn = torch.softmax(attn, dim=-1)
            # mean: b, n_c, c
            mean = torch.bmm(attn, style_flat)
            # std: b, n_c, c
            std = torch.sqrt(torch.relu(torch.bmm(attn, style_flat ** 2) - mean ** 2))
            # mean, std: b, c, h, w
            mean = mean.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
            std = std.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
            loss_local += self.mse_loss(sc_feats[i], std * mean_variance_norm(c_feats[i]) + mean)
            return loss_local

    def forward(self,i_c,i_s):
        f_c_group = self.get_interal_feature(i_c)
        f_s_group = self.get_interal_feature(i_s)
        f_c1 = f_c_group[2]
        f_s1 = f_s_group[2]
        FFC1=torch.fft.fft2(f_c1)
        FFS1=torch.fft.fft2(f_s1)
        FFC1, FFS1 = self.fft_transform1(FFC1, FFS1)
        FFCS1=self.transmodule(FFC1,FFS1)

        IFFCS1=torch.fft.ifft2(FFCS1)
        iffcs1=IFFCS1.real
        f_smean1=f_s1.mean((-2,-1))
        iffcsmean1=iffcs1.mean((-2,-1))
        alpha=(((f_smean1*iffcsmean1)>0)*2.0-1).unsqueeze(2).unsqueeze(3).expand_as(iffcs1)
        iffcs1=iffcs1*alpha

        out_ASA1 = self.ASA3_1(f_c1, f_s1, self.get_key(f_c_group, 2), self.get_key(f_s_group, 2))
        out_ASA_fft1 = torch.cat([iffcs1, out_ASA1], dim=1)
        out_ASA_fft1 = self.conv2d_1(out_ASA_fft1)

        f_c2 = f_c_group[3]
        f_s2 = f_s_group[3]
        FFC2 = torch.fft.fft2(f_c2)
        FFS2 = torch.fft.fft2(f_s2)
        FFC2, FFS2 = self.fft_transform2(FFC2, FFS2)
        FFCS2 = self.transmodule(FFC2, FFS2)

        IFFCS2 = torch.fft.ifft2(FFCS2)
        iffcs2 = IFFCS2.real
        f_smean2 = f_s2.mean((-2, -1))
        iffcsmean2 = iffcs2.mean((-2, -1))
        alpha = (((f_smean2 * iffcsmean2) > 0) * 2.0 - 1).unsqueeze(2).unsqueeze(3).expand_as(iffcs2)
        iffcs2 = iffcs2 * alpha


        out_ASA2 = self.ASA4_1(f_c2, f_s2, self.get_key(f_c_group, 3), self.get_key(f_s_group, 3))
        out_ASA_fft2 = torch.cat([iffcs2, out_ASA2], dim=1)
        out_ASA_fft2 = self.conv2d_2(out_ASA_fft2)

        f_c3 = f_c_group[4]
        f_s3 = f_s_group[4]
        FFC3 = torch.fft.fft2(f_c3)
        FFS3 = torch.fft.fft2(f_s3)
        FFC3, FFS3 = self.fft_transform3(FFC3, FFS3)
        FFCS3 = self.transmodule(FFC3, FFS3)

        IFFCS3 = torch.fft.ifft2(FFCS3)
        iffcs3 = IFFCS3.real
        f_smean3 = f_s3.mean((-2, -1))
        iffcsmean3 = iffcs3.mean((-2, -1))
        alpha = (((f_smean3 * iffcsmean3) > 0) * 2.0 - 1).unsqueeze(2).unsqueeze(3).expand_as(iffcs3)
        iffcs3 = iffcs3 * alpha

        out_ASA3 = self.ASA5_1(f_c3, f_s3, self.get_key(f_c_group, 4), self.get_key(f_s_group, 4))
        out_ASA_fft3 = torch.cat([iffcs3, out_ASA3], dim=1)
        out_ASA_fft3 = self.conv2d_3(out_ASA_fft3)

        out_ASA_fft2 = self.upsample5_1(out_ASA_fft3) + out_ASA_fft2

        d_cs = self.decoder(out_ASA_fft2)
        dcs = d_cs+out_ASA_fft1
        i_cs = self.transnet(dcs)



        f_c_loss = self.get_interal_feature(i_c)#获得内容图特征图组
        f_s_loss = self.get_interal_feature(i_s)#获得风格图特征图组
        f_i_cs_loss = self.get_interal_feature(i_cs)#获得生成图特征图组

        loss_c, loss_s, loss_c_variance, loss_s_mean_std, loss_style_local = 0, 0, 0, 0, 0


        for i in range(1, 5):
            loss_c=  self.calc_content_loss(f_i_cs_loss[i], f_c_loss[i],norm=False)
            loss_c_variance = self.calc_content_loss_variance(f_i_cs_loss[i], f_c_loss[i])


        for i in range(1, 5):
            loss_s += self.calc_style_loss(f_i_cs_loss[i], f_s_loss[i])
            loss_s_mean_std += self.calc_style_loss_gloal_mean_std(f_i_cs_loss[i], f_s_loss[i])

        loss_style_local = self.calc_style_loss_local_mean_std(f_i_cs_loss, f_s_loss, f_c_loss)
        return loss_c, loss_s, i_cs, loss_c_variance, loss_s_mean_std, loss_style_local#, i_cs
