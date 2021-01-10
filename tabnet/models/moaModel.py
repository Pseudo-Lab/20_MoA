import torch
import torch.nn as nn

from models.module import LinearBn


class moaModel(nn.Module):
    def __init__(self, config, num_g, num_c, num_cate,
                 classes=206, non_classes=402):
        super().__init__()
        self.g_enc = LinearBn(num_g, config.hidden_size, dropout=config.dropout)
        self.c_enc = LinearBn(num_c, config.hidden_size, dropout=config.dropout)
        self.cate_enc = LinearBn(num_cate, config.hidden_size, dropout=config.dropout)

        self.layer1 = LinearBn(num_g + num_c, config.hidden_size, dropout=config.dropout)
        self.layer2 = LinearBn(config.hidden_size, config.hidden_size, dropout=config.dropout)
        self.clf = nn.Linear(config.hidden_size, classes)

    def forward(self, g_x, c_x, cate_x):
        # g_x = self.g_enc(g_x)
        # c_x = self.c_enc(c_x)
        # cate_x = self.cate_enc(cate_x)
        embed = torch.cat([g_x, c_x], dim=-1)

        output = self.layer1(embed)
        output = self.layer2(output)
        output = self.clf(output)

        return output