import torch
from torch import nn
import torch.nn.functional as F


class MocoLoss(nn.Module):

    def __init__(self, opts):
        super(MocoLoss, self).__init__()
        self.model = self.__load_model(opts)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @staticmethod
    def __load_model(opts):
        import torchvision.models as models
        model = models.__dict__["resnet50"]()
        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        checkpoint = torch.load(opts.moco_path, map_location="cpu")
        state_dict = checkpoint['state_dict']
        # rename moco pre-trained keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        # remove output layer
        model = nn.Sequential(*list(model.children())[:-1]).cuda()
        return model

    def extract_feats(self, x):
        x = F.interpolate(x, size=224)
        x_feats = self.model(x)
        x_feats = nn.functional.normalize(x_feats, dim=1)
        b, c, _h, _w = x_feats.shape
        x_feats = x_feats.reshape([b, c])
        return x_feats

    def forward(self, y_hat, x):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_hat_feats = self.extract_feats(y_hat)
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_input = y_hat_feats[i].dot(x_feats[i])
            loss += 1 - diff_input
            count += 1
        return loss / count
