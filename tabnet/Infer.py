import os
from tqdm import tqdm
import numpy as np
import torch


class Infer:
    def __init__(self, model, loader, test_config):
        self.config = test_config
        self.device = self.config.device
        self.loader = loader
        self.model = model.to(self.device)

    def inference_data(self, g_x, c_x, cate_x):
        with torch.no_grad():
            preds = torch.sigmoid(self.model(g_x.to(self.config.device),
                                             c_x.to(self.config.device),
                                             cate_x.to(self.config.device))).cpu().numpy()
        return preds

    def inference(self):
        pred = []
        for g_x, c_x, cate_x in self.loader:
            pred.append(self.inference_data(g_x, c_x, cate_x))
        pred = np.concatenate(pred, axis=0)

        # Todo
        if self.config.tta:
            pass
        return pred

    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.config.device)
        self.model.eval()
