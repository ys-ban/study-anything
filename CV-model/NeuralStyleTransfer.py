import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import torch

from matplotlib.pyplot import plt


import torchvision.transforms as transforms


from PIL import Image
from collections import OrderedDict


import os


import segmentation_models_pytorch as smp


#######################
#   F_l : the form of M_l * N_l
#   where M_l = H_L * W_l and N_l is the number of layers of F_l
#   note that M_l depends on the shape of input images but N_l does not
#######################


# Gram matrix is required for a style loss
class GramMatrix(nn.Module):
    def forward(self, inputs):
        #######################
        # N : batch_size
        # C : the number of channels
        # H & W : height and width
        # bmm is matrix multiplication for a chunk of matricies
        #######################
        N, C, H, W = inputs.shape
        flatten_inputs = inputs.view(N, C, H*W)
        # (N, C, H*W) * (N, H*W, C) -> (N, C, C)
        outputs = torch.bmm(flatten_inputs, flatten_inputs.transpose(1, 2))# get an unnormalized gram matrix
        outputs.div_(H*W)# normalize the gram matrix
        return outputs


# content loss
class ContentLoss(nn.Module):
    def forward(self, f_hat, f_content):
        N, C, H, W = f_hat.shape
        SE = torch.pow(f_hat-f_content, 2)
        SE = torch.sum(SE)
        SE.div_(C*H*W)
        return SE


# style loss
class StyleLoss(nn.Module):
    def forward(self, f_hat, g_style):
        N, C, H, W = f_hat.shape
        gram_f_hat = GramMatrix()(f_hat)# (N, C, H, W) -> (N, C, C)
        SE = torch.pow(gram_f_hat-g_style, 2)
        SE = torch.sum(SE)
        SE.div_(4*C*C)
        return SE


class TransferNetwork(nn.Module):
    def __init__(self, encoder_name, encoder_weight, image_dir = "./", content_loss_weight = None, alpha = 1., beta = 1.):
        super(TransferNetwork, self).__init__()
        self.encoder_name = encoder_name
        self.encoder_weight = encoder_weight
        self.encoder = smp.Unet(
                            encoder_name = self.encoder_name,
                            encoder_weights = self.encoder_weight
                            ).encoder
        self.encoder.requires_grad_(False)
        self.filters = self.encoder.out_channels
        self.image_dir = image_dir
        self.img_size = None
        self.img_content = None
        self.img_style = None
        self.content_features = None
        self.style_gram = None
        self.prep = None
        self.postpa = None
        self.postpb = None
        self.style_layers = None
        self.content_layers = None
        self.loss_layers = None

        print("#"*50)
        print(f"encoder name : {encoder_name}\nencoder_weight : {encoder_weight}\nthe number of out-channels : {len(self.filters)}")
        self.get_feature_info(512)
        print("#"*50)
    
    
    def get_feature_info(self, img_size=None):
        if img_size is None:
            assert self.img_size is not None, "img_size is missing."
            img_size = self.img_size
        
        with torch.no_grad():
            tmp_device = "cuda" if torch.cuda.is_available() else "cpu"
            x_test = torch.randn((1, 3, img_size, img_size)).to(tmp_device)
            self.encoder.to(tmp_device)
            out_features = self.encoder(x_test)
            print(f"for the case of input size {img_size} : ")
            for i in range(len(self.filters)):
                print(f"\tthe shape of feature {i} : {out_features[i].shape}")
    
    def set_images(self, param_dict):
        assert isinstance(param_dict, dict), "input is not a dictionary."
        assert ("img_content" in param_dict.keys()) or ("img_style" in param_dict.keys()), "key error."
        if "img_content" in param_dict.keys():
            self.img_content = param_dict["img_content"]
        if "img_content" in param_dict.keys():
            self.img_style = param_dict["img_style"]
    
    def images_info(self):
        print(f"content image : {self.img_content} / style image : {self.img_style}")
    
    def postp(self, tensor):
        assert self.postpa is not None, "self.postpa is None. Please call self.initialize()."
        assert self.postpb is not None, "self.postpb is None. Please call self.initialize()."
        # to clip results in the range [0,1]
        t = self.postpa(tensor)
        t[t>1] = 1
        t[t<0] = 0
        img = self.postpb(t)
        return img

    def intialize(self, img_size=512, style_layers = [3, 4, 5], content_layers = [2]):
        assert self.img_style is not None, "the image for style is None. Please call self.set_images for setting the image information."
        assert self.img_content is not None, "the image for content is None. Please call self.set_images for setting the image information."
        # pre and post processing for images
        self.prep = transforms.Compose([transforms.Resize(img_size),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                                transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                        std=[1,1,1]),
                                transforms.Lambda(lambda x: x.mul_(255)),
                                ])
        self.postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                                transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                        std=[1,1,1]),
                                transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                                ])
        self.postpb = transforms.Compose([transforms.ToPILImage()])
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.loss_layers = style_layers + content_layers

    
    def style_transfer(self, total_iters=1000, save_per=100, log_iter=100, save_dir=None):
        imgs = [Image.open(self.image_dir+self.img_style), Image.open(self.image_dir + self.img_content)]
        imgs_torch = [self.prep(img) for img in imgs]
        if torch.cuda.is_available():
            imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
        else:
            imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
        style_image, content_image = imgs_torch
        opt_img = Variable(content_image.data.clone(), requires_grad=True)

        loss_fns = [StyleLoss()] * len(self.style_layers) + [ContentLoss()] * len(self.content_layers)
        if torch.cuda.is_available():
            loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
        
        #these are good weights settings:
        style_weights = [1e3 for _ in range(len(self.style_layers))]
        content_weights = [1e-0 for _ in range(len(self.content_layers))]
        weights = style_weights + content_weights

        #compute optimization targets
        style_targets = [GramMatrix()(A).detach() for i, A in enumerate(self.forward(style_image)) if i in self.style_layers]
        content_targets = [A.detach() for i, A in enumerate(self.forward(content_image)) if i in self.content_layers]
        targets = style_targets + content_targets
        if save_dir is None:
            save_dir = self.image_dir
        optimizer = optim.LBFGS([opt_img])
        n_iter = [0]
        show_iter = [log_iter]

        while n_iter[0] <= total_iters:
            def closure():
                optimizer.zero_grad()
                out = self.forward(opt_img)
                layer_01 = [out[i] for i in self.style_layers]
                layer_02 = [out[i] for i in self.content_layers]
                layer_losses = layer_01 + layer_02
                layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(layer_losses)]
                loss = sum(layer_losses)
                loss.backward()
                n_iter[0] += 1
                #print loss
                if n_iter[0]%show_iter[0] == (log_iter-1):
                    print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.item()))
                    #print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
                return loss
            
            optimizer.step(closure)
            
        #display result
        out_img = self.postp(opt_img.data[0].cpu().squeeze())
        plt.imshow(out_img)
        gcf().set_size_inches(10,10)

        
    def forward(self, x):
        return self.encoder(x)

