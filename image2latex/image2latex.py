import torch
from torch import nn, Tensor

from data.text import Text, Text100k

from image2latex.convwithpe import ConvEncoderWithPE
from image2latex.decoder import Decoder


class Image2Latex(nn.Module):
    def __init__(
        self,
        n_class: int,
        enc_dim: int = 512,
        enc_type: str = "conv_row_encoder",
        emb_dim: int = 80,
        dec_dim: int = 512,
        attn_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        decode_type: str = "greedy",
        text: Text = None,
        beam_width: int = 5,
        sos_id: int = 1,
        eos_id: int = 2,
    ):
            
        
        super().__init__()
        self.n_class = n_class
        self.encoder = ConvEncoderWithPE(enc_dim=enc_dim)
        enc_dim = self.encoder.enc_dim
        self.num_layers = num_layers
        self.decoder = Decoder(
            n_class=n_class,
            emb_dim=emb_dim,
            dec_dim=dec_dim,
            enc_dim=enc_dim,
            attn_dim=attn_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            sos_id=sos_id,
            eos_id=eos_id,
        )
        self.init_h = nn.Linear(enc_dim, dec_dim)
        self.init_c = nn.Linear(enc_dim, dec_dim)
        assert decode_type in ["greedy", "beamsearch"]
        self.decode_type = decode_type
        self.text = text
        self.beam_width = beam_width

    def init_decoder_hidden_state(self, V: Tensor):
        """
            return (h, c)
        """
        # V has size (bs, -1, d)
        
        encoder_mean = V.mean(dim=1)
        h = torch.tanh(self.init_h(encoder_mean))
        c = torch.tanh(self.init_c(encoder_mean))
        return h, c

    def forward(self, x: Tensor, y: Tensor, y_len: Tensor):
        encoder_out = self.encoder(x)

        hidden_state = self.init_decoder_hidden_state(encoder_out)

        predictions = []
        for t in range(y_len.max().item()):
            dec_input = y[:, t].unsqueeze(1)
            out, hidden_state = self.decoder(dec_input, encoder_out, hidden_state)
            predictions.append(out.squeeze(1))

        predictions = torch.stack(predictions, dim=1)
        return predictions

    def decode(self, x: Tensor, max_length: int = 150):
        predict = self.decode_beam_search(x, max_length)
        return self.text.int2text(predict)

    def decode_beam_search(self, x: Tensor, max_length: int = 150):
        """
        使用beam search算法进行解码
        :param x: 输入张量
        :param max_length: 最大长度，默认为150
        :return: 解码结果
        """

        encoder_out = self.encoder(x)  # 使用encoder进行编码
        bs = encoder_out.size(0)  # 批次大小为1

        hidden_state = self.init_decoder_hidden_state(encoder_out)  # 初始化解码器的隐藏状态

        list_candidate = [
            ([self.decoder.sos_id], hidden_state, 0)  # 初始候选列表只有一个元素，输入为起始符(self.decoder.sos_id)，隐藏状态为初始化隐藏状态，概率为0
        ]

        for t in range(max_length):
            new_candidates = []  # 创建一个新的候选列表
            for inp, state, log_prob in list_candidate:  # 遍历当前候选列表中的每个元素
                y = torch.LongTensor([inp[-1]]).view(bs, -1).to(device=x.device)  # 创建一个形状为(bs, 1)的张量，其中bs为批次大小，值为输入序列中最后一个词元
                out, hidden_state = self.decoder(y, encoder_out, state)  # 使用decoder进行解码，输入为上一个词元、编码器输出和当前隐藏状态，得到解码输出和新的隐藏状态

                topk = out.topk(self.beam_width)  # 获取解码输出中概率最高的self.beam_width个值及其索引

                new_log_prob = topk.values.view(-1).tolist()  # 将self.beam_width个概率值转换为列表

                new_idx = topk.indices.view(-1).tolist()  # 将self.beam_width个索引转换为列表
                for val, idx in zip(new_log_prob, new_idx):  # 遍历新的概率值和索引
                    new_inp = inp + [idx]  # 当前输入序列为上一个输入序列加上新的索引
                    new_candidates.append((new_inp, hidden_state, log_prob + val))  # 将新的候选序列（输入序列、隐藏状态、概率）添加到新的候选列表中

            new_candidates = sorted(new_candidates, key=lambda x: x[2], reverse=True)  # 根据概率对新的候选列表进行降序排序
            list_candidate = new_candidates[: self.beam_width]  # 保留概率最高的self.beam_width个候选序列

        return list_candidate[0][0]  # 返回解码结果，即概率最高的候选序列的第一个输入词元
