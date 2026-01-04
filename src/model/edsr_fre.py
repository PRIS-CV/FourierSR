from model import common
import torch.nn as nn
import torch

print('this is edsr fre')


def make_model(args, parent=False):
    return EDSR(args)


def report_model(model):
    n_parameters = 0
    for p in model.parameters():
        n_parameters += p.nelement()
    print("Model  with {} parameters".format(n_parameters))


####################################################################################################
import matplotlib.pyplot as plt
import numpy as np
def visualize_feature_map(img_batch):
    #img_batch = F.interpolate(img_batch, size=(img_batch.shape[1]*4,img_batch.shape[2]*4), mode='bilinear', align_corners=False)
    img_batch = img_batch.reshape(img_batch.shape[1:])
    feature_map = np.squeeze(img_batch, axis=0)
    feature_map = img_batch.cpu().numpy()

    print(feature_map.shape)

    pmax = np.max(feature_map)
    pmin = np.min(feature_map)
    feature_map = ((feature_map - pmin) / (pmax - pmin + 0.000001)) * 255
    feature_map = feature_map.astype(np.uint8)

    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[0]
    row, col = get_row_col(num_pic)
    # print(num_pic)
    for i in range(0, num_pic):
        #plt.figure()   ## singer channels img need #
        print(i)
        feature_map_split = feature_map[i, :, :]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        plt.axis('off')
        ###################  channels imgs   #####################
        w=feature_map_split.shape[0]
        h=feature_map_split.shape[1]
        dpi=100
        fig=plt.figure(figsize=(w,h))
        axes=fig.add_axes([0,0,1,1])
        axes.set_axis_off()
        axes.imshow(feature_map_split)
        fig.savefig("/home/liwenjie/wenjieli/SR/PCS-master/src/features/{}.png".format(i))  #color fig

    #     #cv2.imwrite("/home2/wenjieli/MsDNN_LWJ_Trans1/RB_img1/train{}.png".format(i), feature_map_split)   #gray fig
    #plt.savefig('CIAM.png')
    # plt.show()

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
####################################################################################################


class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        m_body = [
            common.ResBlock_fre(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        # visualize_feature_map(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
