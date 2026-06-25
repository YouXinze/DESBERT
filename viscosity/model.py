import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import nn
from transformers import RobertaConfig, RobertaModel


class CNN(nn.Module):
    """Network for fine-tuning on IL properties datasets"""

    def __init__(self,
                 dropout,
                 embed_size,
                 output_size=1,
                 num_filters=(100, 200, 200, 200, 200, 100, 100),
                 ngram_filter_sizes=(1, 2, 3, 4, 5, 6, 7),
                 IL_num_filters=(100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160),
                 IL_ngram_filter_sizes=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15)):
        super(CNN, self).__init__()

        self.num_filters = num_filters
        self.IL_num_filters = IL_num_filters

        self.IL_textcnn = nn.ModuleList([nn.Conv1d(in_channels=embed_size, out_channels=nf, kernel_size=ks)
                                         for nf, ks in zip(IL_num_filters, IL_ngram_filter_sizes)])
        self.output = nn.Linear(sum(IL_num_filters), embed_size)

    def forward(self, IL_src_nd):
        IL_encoded = IL_src_nd.permute(1, 2, 0)

        IL_textcnn_out = [F.relu(conv(IL_encoded)) for conv in self.IL_textcnn]
        IL_textcnn_out = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in IL_textcnn_out]  # Max pooling
        IL_textcnn_out = torch.cat(IL_textcnn_out, 1)  # Concatenate all the pooled features
        # input_vecs = torch.cat((IL_textcnn_out, T.view(-1, 1), P.view(-1, 1)), dim=1)
        input_vecs = IL_textcnn_out
        out = self.output(input_vecs.float())

        # out = self.output(IL_encoded[:,0,:].float())

        return out


class ILBERT(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float):
        super().__init__()
        self.model_type = 'RoBERTa'

        config = RobertaConfig(
            vocab_size=ntoken,  # 词汇表大小
            hidden_size=d_model,  # 隐藏层维度
            num_hidden_layers=nlayers,  # Transformer层数
            num_attention_heads=nhead,  # 注意力头数
            intermediate_size=d_hid,  # 中间层维度
            hidden_dropout_prob=dropout,  # 隐藏层dropout
            attention_probs_dropout_prob=dropout,
            output_attentions=True,  # 确保返回注意力权重
        )
        self.roberta = RobertaModel(config)
        self.CNN = CNN(embed_size=d_model, dropout=dropout)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model + 1
                      , d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, input):
        """
        Args:
            input: 输入数据，格式为(src, labels),其中src的形状应为[batch_size, seq_len]
        Returns:
            output: 模型输出
            attentions: 注意力权重
        """

        x1, x2, mol1, mol2,HA1, HD1, HA2, HD2, T = input

        # 生成attention_mask，对于src中不为0的位置，mask值为1；否则为0
        attention_mask1 = (x1 != 0).long()
        attention_mask2 = (x2 != 0).long()
        outputs1 = self.roberta(input_ids=x1.long(), attention_mask=attention_mask1)
        outputs2 = self.roberta(input_ids=x2.long(), attention_mask=attention_mask2)
        last_hidden_states1 = outputs1.last_hidden_state*mol1.view(-1,1,1)
        last_hidden_states2 = outputs2.last_hidden_state*mol2.view(-1,1,1)
        last_hidden_states = torch.cat((last_hidden_states1, last_hidden_states2),dim=1)
        last_hidden_states = last_hidden_states.permute(1, 0, 2)

        mixture_features = self.CNN(last_hidden_states)

        output = torch.cat([mixture_features, T.view(-1, 1)], dim=1)
        output = self.pred_head(output)

        return output


