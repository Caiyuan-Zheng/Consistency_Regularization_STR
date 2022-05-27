import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_hid)

    def forward(self, x):
        output = self.w_1(x.transpose(1, 2)).transpose(1, 2)
        output = F.relu(self.layer_norm(output))
        output = self.w_2(output.transpose(1, 2)).transpose(1, 2)
        return output


class Attention(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_class,
                 num_char_embeddings=256,
                 projection_type="pff"):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size,
                                            num_char_embeddings)
        self.hidden_size = hidden_size
        self.num_class = num_class
        if projection_type == "pff":
            self.projection_head = PositionwiseFeedForward(
                hidden_size, hidden_size // 4)
        elif projection_type == 'linear':
            self.projection_head = nn.Linear(hidden_size, hidden_size)
        else:
            assert False, 'no projection type name %s!' % projection_type

        self.generator = nn.Linear(hidden_size, num_class)
        self.char_embeddings = nn.Embedding(num_class, num_char_embeddings)

    def forward(self,
                batch_H,
                text,
                is_train=True,
                batch_max_length=25,
                use_project=False,
                return_local_feat=False):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x num_class]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [SOS] token. text[:, 0] = [SOS].
        output: probability distribution at each step [batch_size x num_steps x num_class]
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [EOS] at end of sentence.

        output_hiddens = torch.cuda.FloatTensor(batch_size, num_steps,
                                                self.hidden_size).fill_(0)
        local_feats = torch.cuda.FloatTensor(batch_size, num_steps,
                                             self.hidden_size).fill_(0)
        hidden = (torch.cuda.FloatTensor(batch_size,
                                         self.hidden_size).fill_(0),
                  torch.cuda.FloatTensor(batch_size,
                                         self.hidden_size).fill_(0))

        if is_train:
            for i in range(num_steps):
                char_embeddings = self.char_embeddings(text[:, i])
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_embeddings : f(y_{t-1})
                hidden, alpha, context = self.attention_cell(
                    hidden, batch_H, char_embeddings)
                output_hiddens[:, i, :] = hidden[
                    0]  # LSTM hidden index (0: hidden, 1: Cell)
                local_feats[:, i, :] = context
            if use_project:
                output_hiddens = self.projection_head(output_hiddens)
            probs = self.generator(output_hiddens)
            if return_local_feat:
                return probs, local_feats
            else:
                return probs
        else:
            targets = text[0].expand(
                batch_size)  # should be fill with [SOS] token

            probs = torch.cuda.FloatTensor(batch_size, num_steps,
                                           self.num_class).fill_(0)
            confident_list = torch.cuda.FloatTensor(batch_size).fill_(0)
            for i in range(num_steps):
                char_embeddings = self.char_embeddings(targets)
                hidden, alpha, _ = self.attention_cell(hidden, batch_H,
                                                       char_embeddings)
                part1, part2 = hidden
                if use_project:
                    part1 = self.projection_head(part1)
                hidden = (part1, part2)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                probs_step = F.log_softmax(probs_step, dim=-1)
                scores_step, next_input = probs_step.max(1)
                confident_list += scores_step
                targets = next_input

            return probs  # batch_size x num_steps x num_class


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size,
                             hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_embeddings):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(
            torch.tanh(batch_H_proj +
                       prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1),
                            batch_H).squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat(
            [context, char_embeddings],
            1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha, context
