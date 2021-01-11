import copy
from urllib.request import urlopen
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision.transforms import transforms
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 256

loader = transforms.Compose([
    transforms.Resize([imsize, imsize]),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()

    features = input.view(a * b, c * d)

    G = torch.mm(features, features.t())

    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()

        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class Predictor:
    def __init__(self):
        self.model = nn.Sequential()

    def get_image_predict(self, img_path='img_path.jpg', option="1"):
        img_tensor = image_loader(img_path)

        if option == "1":
            style_img = image_loader(urlopen('https://avatars.mds.yandex.net/get-pdb/988157/6f568a41-ed9d-4291-8715-5380a0e40cf9/s1200?webp=false'))
        if option == "2":
            style_img = image_loader(urlopen('https://avatars.mds.yandex.net/get-zen_doc/3300410/pub_5eeb13472ab5ff06f3a8377d_5eeb19035b22211b8e3c9c70/scale_1200'))
        if option == "3":
            style_img = image_loader(urlopen('https://media.leverans.ru/product_images/rostov/sushi-party/e7f4bef15d8e617faacebe10f0fc22a9.jpg'))
        if option == "4":
            style_img = image_loader(urlopen('https://avatars.mds.yandex.net/get-zen_doc/1587994/pub_5ea2d651ea9eca5d5696cc28_5ea2d6bdf37bf573d64d75a0/scale_1200'))
        if option == "5":
            style_img = image_loader(urlopen('https://avatars.mds.yandex.net/get-pdb/226447/b2e02377-03a9-4472-a8df-3170c8f9416e/s1200?webp=false'))
        if option == "6":
            style_img = image_loader(urlopen('https://avatars.mds.yandex.net/get-zen_doc/235990/pub_5c85f40073c88f00b4414aa0_5c85fa5ebc0ee200b3eae899/scale_1200'))
        if option == "7":
            style_img = image_loader(urlopen('https://avatars.mds.yandex.net/get-pdb/214107/2fdcd853-3720-4e6a-a53d-6c263ff6460e/s1200?webp=false'))
        if option == "8":
            style_img = image_loader(urlopen('https://avatars.mds.yandex.net/get-pdb/1340633/a5776848-97d5-4ac9-a200-c4df01a4257b/s1200?webp=false'))
        if option == "9":
            style_img = image_loader(urlopen('https://avatars.mds.yandex.net/get-pdb/812271/5bd681cb-4e16-4354-8c93-d8a46272a4b2/s1200?webp=false'))
        if option == "10":
            style_img = image_loader(urlopen('http://wikiooimg.wikioo.org/ADC/Art-ImgScreen-1.nsf/O/A-8XXKE8/$FILE/Alexej_georgewitsch_von_jawlensky-landscape_oberstdorf.Jpg'))
        if option == "11":
            style_img = image_loader(urlopen('https://sun9-56.userapi.com/c625316/v625316749/3832e/v7ZFuGBxbSg.jpg'))

        cnn = models.vgg19(pretrained=True).features.to(device).eval()

        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        content_layers_default = ['conv_4']
        style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                       style_img, content_img,
                                       content_layers=content_layers_default,
                                       style_layers=style_layers_default):
            cnn = copy.deepcopy(cnn)

            # normalization module
            normalization = Normalization(normalization_mean, normalization_std).to(device)

            # just in order to have an iterable access to or list of content/syle
            # losses
            content_losses = []
            style_losses = []

            # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
            # to put in modules that are supposed to be activated sequentially
            model = nn.Sequential(normalization)

            i = 0  # increment every time we see a conv
            for layer in cnn.children():
                if isinstance(layer, nn.Conv2d):
                    i += 1
                    name = 'conv_{}'.format(i)
                elif isinstance(layer, nn.ReLU):
                    name = 'relu_{}'.format(i)
                    # The in-place version doesn't play very nicely with the ContentLoss
                    # and StyleLoss we insert below. So we replace with out-of-place
                    # ones here.
                    layer = nn.ReLU(inplace=False)
                elif isinstance(layer, nn.MaxPool2d):
                    name = 'pool_{}'.format(i)
                elif isinstance(layer, nn.BatchNorm2d):
                    name = 'bn_{}'.format(i)
                else:
                    raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

                model.add_module(name, layer)

                if name in content_layers:
                    # add content loss:
                    target = model(content_img).detach()
                    content_loss = ContentLoss(target)
                    model.add_module("content_loss_{}".format(i), content_loss)
                    content_losses.append(content_loss)

                if name in style_layers:
                    # add style loss:
                    target_feature = model(style_img).detach()
                    style_loss = StyleLoss(target_feature)
                    model.add_module("style_loss_{}".format(i), style_loss)
                    style_losses.append(style_loss)

            # now we trim off the layers after the last content and style losses
            for i in range(len(model) - 1, -1, -1):
                if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                    break

            model = model[:(i + 1)]

            return model, style_losses, content_losses

        input_img = img_tensor.clone()

        def get_input_optimizer(input_img):
            # this line to show that input is a parameter that requires a gradient
            optimizer = torch.optim.LBFGS([input_img.requires_grad_()])
            return optimizer

        def run_style_transfer(cnn, normalization_mean, normalization_std,
                               content_img, style_img, input_img, num_steps=100,
                               style_weight=500000, content_weight=5):

            model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                             normalization_mean, normalization_std,
                                                                             style_img, content_img)
            optimizer = get_input_optimizer(input_img)

            print('Optimizing..')
            run = [0]
            while run[0] <= num_steps:

                def closure():
                    # correct the values of updated input image
                    input_img.data.clamp_(0, 1)

                    optimizer.zero_grad()
                    model(input_img)
                    style_score = 0
                    content_score = 0

                    for sl in style_losses:
                        style_score += sl.loss
                    for cl in content_losses:
                        content_score += cl.loss

                    style_score *= style_weight
                    content_score *= content_weight

                    loss = style_score + content_score
                    loss.backward()

                    run[0] += 1

                    return style_score + content_score

                optimizer.step(closure)
                if run[0] % 10 == 0:
                    yield input_img.cpu().data

            # a last correction...
            input_img.data.clamp_(0, 1)

            yield input_img

        for output in run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    img_tensor, style_img, input_img):

            yield output
            print("!")
        print('training finished')
