import torch
from torch.nn import Module, Sequential
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, MaxPool2d, AvgPool1d, Dropout2d
from torch.nn import ReLU, Sigmoid


class UNet(Module):

    def __init__(self, batch_size, num_channels=3, feat_channels=[48, 256, 256, 512, 1024], residual='conv'):
        # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

        super(UNet, self).__init__()
        
        self.batch_size = batch_size
        
        # Encoder downsamplers
        self.pool1 = MaxPool2d((2, 2))
        self.pool2 = MaxPool2d((2, 2))
        self.pool3 = MaxPool2d((2, 2))
        self.pool4 = MaxPool2d((2, 2))

        # Encoder convolutions
        self.conv_blk1 = Conv2D_Block(num_channels, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv2D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv2D_Block(feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv2D_Block(feat_channels[2], feat_channels[3], residual=residual)
        self.conv_blk5 = Conv2D_Block(feat_channels[3], feat_channels[4], residual=residual)

        # Decoder convolutions
        self.dec_conv_blk4 = Conv2D_Block(2 * feat_channels[3], feat_channels[3], residual=residual)
        self.dec_conv_blk3 = Conv2D_Block(2 * feat_channels[2], feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv2D_Block(2 * feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv2D_Block(2 * feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers
        self.deconv_blk4 = Deconv2D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv2D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv2D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv2D_Block(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        self.one_conv = Conv2d(feat_channels[0], 1, kernel_size=1, stride=1, padding=0, bias=True)
        
        # corrisponde alla quantità diversa di tipi di oggetti che il modello è in grado di segmentare 
        # es. mais e cocco (2) oppure mais cocco e foraggiere (3)
        num_classes = 18 # magari va portato nel costruttore, potrebbe variare in base al problema
        # creo una conv per avere il numero di classi desiderato
        self.final_conv = torch.nn.Conv2d(48, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        # Activation function
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # Encoder part
        # remove everything but rgb
        x = torch.squeeze(x)
        if x.dim() < 4:
            x = torch.unsqueeze(x,0)
        x = x[:self.batch_size, 0:3,:48,:48]

        #x = torch.permute(x, (1,0,2,3))
        #x = torch.nn.functional.pad(x,(0, 2), mode='constant', value=0.0)
        #print(x.shape)
        #x = torch.permute(x, (0,1,4,3,2))
        #print(x.shape)
        x1 = self.conv_blk1(x)
        #print("x1", x1.shape)
        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)
        #print("x4", x2.shape)
        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)
        #print("x3", x3.shape)
        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)
        #print("x4", x4.shape)
        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)

        # Decoder part
        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        test2 = self.deconv_blk3(d_high4)
       # print(test2.shape)
       # print(x3.shape)
        d3 = torch.cat([test2, x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d_high3 = Dropout2d(p=0.5)(d_high3)
        #print("d_high3", d_high3.shape)

        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d_high2 = Dropout2d(p=0.5)(d_high2)
        #print("d_high2", d_high2.shape)
        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)
        #print("d_high1", d_high1.shape)        
        
        #conv_out = self.one_conv(d_high1)
        # la sua shape è (b, 1, 32, 48, 48)
        #print(conv_out.shape)
        # 1 è come avere [[v1,v2...]] che è equivalente a [v1,v2...]
        # possiamo eliminare la dimensione con il comando squeeze
        #conv_out = conv_out.squeeze(1)
        #print(conv_out.shape)
        # (b, 32, 48, 48)
        # uso una conv per portarmi a (b, num_classes, 48, 48)
        
        #d_high1 = torch.permute(d_high1, (1,0,2,3))
        out = self.final_conv(d_high1)
        print("out: ", out.shape)
        seg = self.sigmoid(out)

        return seg


class Conv2D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):

        super(Conv2D_Block, self).__init__()

        self.conv1 = Sequential(
            Conv2d(inp_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm2d(out_feat),
            ReLU())

        self.conv2 = Sequential(
            Conv2d(out_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm2d(out_feat),
            ReLU())

        self.residual = residual
        
        if self.residual is not None:
            self.residual_upsampler = Conv2d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):

        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class Deconv2D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=2, padding=1):
        super(Deconv2D_Block, self).__init__()

        self.deconv = Sequential(
            ConvTranspose2d(inp_feat, out_feat, kernel_size=kernel,
                            stride=stride, padding=padding, output_padding=1, bias=True),
            ReLU())

    def forward(self, x):
        return self.deconv(x)


class ChannelPool2d(AvgPool1d):

    def __init__(self, kernel_size, stride, padding):
        super(ChannelPool2d, self).__init__(kernel_size, stride, padding)
        self.pool_1d = AvgPool1d(self.kernel_size, self.stride, self.padding, self.ceil_mode)

    def forward(self, inp):
        n, c, d, w, h = inp.size()
        inp = inp.view(n, c, d * w * h).permute(0, 2, 1)
        pooled = self.pool_1d(inp)
        c = int(c / self.kernel_size[0])
        return inp.view(n, c, d, w, h)