import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image,ImageDraw
import os

print('Loading model..')
net = RetinaNet()
net.load_state_dict(torch.load('checkpoint/ckpt.pth')['net'])
net.eval()
net.cuda()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
the_classes=[c.strip() for c in open('data/voc.names').readlines()]
encoder = DataEncoder()
for i in os.listdir('image'):
    print('Loading image %s...'%i)
    img = Image.open(os.path.join('image',i))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    w = h = 384
    img = img.resize((w,h))
    # img.save(os.path.join('output', i))
    print('Predicting..')
    x = transform(img)
    x = x.unsqueeze(0)
    x =x.cuda()
    # with torch.no_grad():
    loc_preds, cls_preds = net(x)

    print('Decoding..')

    boxes, labels,scores = encoder.decode(loc_preds.data, cls_preds.data, (w,h))

    draw = ImageDraw.Draw(img)
    for index,box in enumerate(boxes):
        draw.rectangle(list(box), outline='red')
        # draw.text(list(box)[:2],text=the_classes[int(labels[index])])
    img.save(os.path.join('output',i))

