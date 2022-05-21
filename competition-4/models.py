import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import math

from torch.nn import LSTM, GRU


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.batch_first = batch_first
        if batch_first:
            pe = pe.unsqueeze(0)
        else:
            pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MouseEncoder(nn.Module):
    """
    emb_d_total : emb_d + 5
    """
    def __init__(self, emb_d, n_head=2, n_hid=128, n_layers=3, dropout=0.5):
        super(MouseEncoder, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.mouse_embeddings = nn.Embedding(3, emb_d)
        self.pos_encoder = PositionalEncoding(emb_d+5, dropout)
        encoder_layers = TransformerEncoderLayer(emb_d+5, n_head, n_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.emb_d = emb_d
        self.emb_d_total = emb_d+5
        self.m_zero = nn.Parameter(torch.zeros((1, emb_d + 5), dtype=torch.float))
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.mouse_embeddings.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src):
        # src : batch_first=True
        m_event_emb = self.mouse_embeddings(src["eventType"])
        src_ti = (src["timeInterval"].unsqueeze(-1) / 1000.).clamp(0, 10)
        src_absX = src["absX"].unsqueeze(-1) / 1200
        src_absY = src["absY"].unsqueeze(-1) / 600
        src_diffX = src["diffX"].unsqueeze(-1).clamp(-256, 256) / 32
        src_diffY = src["diffY"].unsqueeze(-1).clamp(-256, 256) / 32
        m_inputs = torch.cat((m_event_emb, src_ti, src_absX, src_absY, src_diffX, src_diffY), -1)

        m_zero = self.m_zero.unsqueeze(0).expand(m_inputs.size(0), -1, -1)
        m_inputs = torch.cat((m_inputs, m_zero), 1)
        
        m_inputs = m_inputs * math.sqrt(self.emb_d_total)
        # src : batch_first=False
        m_inputs = m_inputs.permute([1, 0, 2])
        m_inputs = self.pos_encoder(m_inputs)
        output = self.transformer_encoder(m_inputs)
        # src : batch_first=True
        return output.permute([1, 0, 2])


class KeyIdEncoder(nn.Module):
    """
    emb_d_total : emb_d + emb_d_keycode + 1
    """
    def __init__(self, emb_d, emb_d_keycode, n_head=2, n_hid=128, n_layers=3, dropout=0.5):
        super(KeyIdEncoder, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.ki_embeddings = nn.Embedding(2, emb_d)
        self.keycode_embeddings = nn.Embedding(2257, emb_d_keycode)
        self.pos_encoder = PositionalEncoding(emb_d+emb_d_keycode+1, dropout)
        encoder_layers = TransformerEncoderLayer(emb_d+emb_d_keycode+1, n_head, n_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.emb_d = emb_d
        self.emb_d_keycode = emb_d_keycode
        self.emb_d_total = emb_d+emb_d_keycode+1
        self.ki_zero = nn.Parameter(torch.zeros((1, emb_d + emb_d_keycode + 1), dtype=torch.float))
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.ki_embeddings.weight.data.uniform_(-initrange, initrange)
        self.keycode_embeddings.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src):
        # src : batch_first=True
        ki_event_emb = self.ki_embeddings(src["eventType"])
        keycode_emb = self.keycode_embeddings(src["keyCode"])
        src_ti = (src["timeInterval"].unsqueeze(-1) / 1000.).clamp(0, 1)
        ki_inputs = torch.cat((ki_event_emb, keycode_emb, src_ti), -1)

        ki_zero = self.ki_zero.unsqueeze(0).expand(ki_inputs.size(0), -1, -1)
        ki_inputs = torch.cat((ki_inputs, ki_zero), 1)

        ki_inputs = ki_inputs * math.sqrt(self.emb_d_total)
        # src : batch_first=False
        ki_inputs = ki_inputs.permute([1, 0, 2])
        ki_inputs = self.pos_encoder(ki_inputs)
        output = self.transformer_encoder(ki_inputs)
        # src : batch_first=True
        return output.permute([1, 0, 2])



class KeyPwEncoder(nn.Module):
    """
    emb_d_total : emb_d + 1
    """
    def __init__(self, emb_d, n_head=2, n_hid=128, n_layers=3, dropout=0.5):
        super(KeyPwEncoder, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.kp_embeddings = nn.Embedding(2, emb_d)
        self.pos_encoder = PositionalEncoding(emb_d+1, dropout)
        encoder_layers = TransformerEncoderLayer(emb_d+1, n_head, n_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.emb_d = emb_d
        self.emb_d_total = emb_d+1
        self.kp_zero = nn.Parameter(torch.zeros((1, emb_d + 1), dtype=torch.float))
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.kp_embeddings.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src):
        # src : batch_first=True
        kp_event_emb = self.kp_embeddings(src["eventType"])
        src_ti = (src["timeInterval"].unsqueeze(-1) / 1000.).clamp(0, 1)
        kp_inputs = torch.cat((kp_event_emb, src_ti), -1)

        kp_zero = self.kp_zero.unsqueeze(0).expand(kp_inputs.size(0), -1, -1)
        kp_inputs = torch.cat((kp_inputs, kp_zero), 1)

        kp_inputs = kp_inputs * math.sqrt(self.emb_d_total)
        # src : batch_first=False
        kp_inputs = kp_inputs.permute([1, 0, 2])
        kp_inputs = self.pos_encoder(kp_inputs)
        output = self.transformer_encoder(kp_inputs)
        # src : batch_first=True
        return output.permute([1, 0, 2])

class AbuseDetector(nn.Module):
    def __init__(
        self,
        m_emb_d=11, m_head=2, m_hid=64, m_layers=3,
        ki_emb_d=4, keycode_emb_d=11, ki_head=2, ki_hid=64, ki_layers=3,
        kp_emb_d=15, kp_head=2, kp_hid=64, kp_layers=3, 
        dropout=0.5, classifier_dropout=0.5,
    ):
        super(AbuseDetector, self).__init__()
        self.m_encoder = MouseEncoder(m_emb_d, m_head, m_hid, m_layers, dropout)
        self.ki_encoder = KeyIdEncoder(ki_emb_d, keycode_emb_d, ki_head, ki_hid, ki_layers, dropout)
        self.kp_encoder = KeyPwEncoder(kp_emb_d, kp_head, kp_hid, kp_layers, dropout)
        self.total_dim = self.m_encoder.emb_d_total + self.ki_encoder.emb_d_total + self.kp_encoder.emb_d_total
        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(self.total_dim, 2)
        )
        
    
    def forward(self, inputs, lengths):
        m, ki, kp = inputs
        m_ls, ki_ls, kp_ls = lengths

        batch_size = inputs[0]['timeInterval'].size(0)
        m_encoded = self.m_encoder(m)
        ki_encoded = self.ki_encoder(ki)
        kp_encoded = self.kp_encoder(kp)
        m_state = m_encoded[0:1, int(m_ls[0].item()),:]
        ki_state = ki_encoded[0:1, int(ki_ls[0].item()),:]
        kp_state = kp_encoded[0:1, int(kp_ls[0].item()),:]
        for i in range(1, batch_size):
            m_state = torch.cat(
                [
                    m_state,
                    m_encoded[i:i+1, int(m_ls[i].item()),:]
                ],
                dim = 0
            )
            ki_state = torch.cat(
                [
                    ki_state,
                    ki_encoded[i:i+1, int(ki_ls[i].item()),:]
                ],
                dim = 0
            )
            kp_state = torch.cat(
                [
                    kp_state,
                    kp_encoded[i:i+1, int(kp_ls[i].item()),:]
                ],
                dim = 0
            )
        
        cat_encoded = torch.cat([m_state, ki_state, kp_state], dim=1)
        
        return self.classifier(cat_encoded)


class AbuseDetectorV2(nn.Module):
    def __init__(
        self,
        m_emb_d=11, m_head=2, m_hid=64, m_layers=3,
        ki_emb_d=4, keycode_emb_d=11, ki_head=2, ki_hid=64, ki_layers=3,
        kp_emb_d=15, kp_head=2, kp_hid=64, kp_layers=3, 
        dropout=0.5, classifier_dropout=0.5,
    ):
        super(AbuseDetectorV2, self).__init__()
        self.m_encoder = MouseEncoder(m_emb_d, m_head, m_hid, m_layers, dropout)
        self.ki_encoder = KeyIdEncoder(ki_emb_d, keycode_emb_d, ki_head, ki_hid, ki_layers, dropout)
        self.kp_encoder = KeyPwEncoder(kp_emb_d, kp_head, kp_hid, kp_layers, dropout)
        self.total_dim = self.m_encoder.emb_d_total + self.ki_encoder.emb_d_total + self.kp_encoder.emb_d_total
        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(self.total_dim, self.total_dim*2),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(self.total_dim, 2),
        )
        
    
    def forward(self, inputs, lengths):
        m, ki, kp = inputs
        m_ls, ki_ls, kp_ls = lengths
        batch_size = inputs[0]['timeInterval'].size(0)
        m_encoded = self.m_encoder(m)
        ki_encoded = self.ki_encoder(ki)
        kp_encoded = self.kp_encoder(kp)
        m_state = m_encoded[0:1, int(m_ls[0].item()),:]
        ki_state = ki_encoded[0:1, int(ki_ls[0].item()),:]
        kp_state = kp_encoded[0:1, int(kp_ls[0].item()),:]
        for i in range(1, batch_size):
            m_state = torch.cat(
                [
                    m_state,
                    m_encoded[i:i+1, int(m_ls[i].item()),:]
                ],
                dim = 0
            )
            ki_state = torch.cat(
                [
                    ki_state,
                    ki_encoded[i:i+1, int(ki_ls[i].item()),:]
                ],
                dim = 0
            )
            kp_state = torch.cat(
                [
                    kp_state,
                    kp_encoded[i:i+1, int(kp_ls[i].item()),:]
                ],
                dim = 0
            )
        
        cat_encoded = torch.cat([m_state, ki_state, kp_state], dim=1)
        
        return self.classifier(cat_encoded)


class AbuseDetectorV3(nn.Module):
    def __init__(
        self,
        m_emb_d=11, m_head=2, m_hid=64, m_layers=3,
        ki_emb_d=4, keycode_emb_d=11, ki_head=2, ki_hid=64, ki_layers=3,
        kp_emb_d=15, kp_head=2, kp_hid=64, kp_layers=3, 
        dropout=0.5, classifier_dropout=0.5,
    ):
        super(AbuseDetectorV3, self).__init__()
        self.m_encoder = MouseEncoder(m_emb_d, m_head, m_hid, m_layers, dropout)
        self.ki_encoder = KeyIdEncoder(ki_emb_d, keycode_emb_d, ki_head, ki_hid, ki_layers, dropout)
        self.kp_encoder = KeyPwEncoder(kp_emb_d, kp_head, kp_hid, kp_layers, dropout)
        self.total_dim = self.m_encoder.emb_d_total + self.ki_encoder.emb_d_total + self.kp_encoder.emb_d_total
        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(self.total_dim, 2)
        )
        
    
    def forward(self, inputs):
        m, ki, kp = inputs

        # batch_size = inputs[0]['timeInterval'].size(0)
        m_encoded = self.m_encoder(m)[:,-1,:]
        ki_encoded = self.ki_encoder(ki)[:,-1,:]
        kp_encoded = self.kp_encoder(kp)[:,-1,:]
        cat_encoded = torch.cat([m_encoded, ki_encoded, kp_encoded], dim=1)
        return self.classifier(cat_encoded)
        


def lstm_padded(lstm, inputs, ls):
    ls, idx = ls.sort(descending=True)
    invert_idx = idx.sort()[1]
    assert 0 <= idx.min() <= idx.max() < inputs.size(0), (inputs.size(0), idx.min(), idx.max())
    packed = pack_padded_sequence(inputs[idx], ls.tolist(), batch_first=True)
    _, (hidden, _) = lstm(packed)
    hidden = hidden[-2:, invert_idx]
    hidden = torch.cat((hidden[0], hidden[1]), -1)
    return hidden

def gru_padded(gru, inputs, ls):
    ls, idx = ls.sort(descending=True)
    invert_idx = idx.sort()[1]
    assert 0 <= idx.min() <= idx.max() < inputs.size(0), (inputs.size(0), idx.min(), idx.max())
    packed = pack_padded_sequence(inputs[idx], ls.tolist(), batch_first=True)
    _, hidden = gru(packed)
    hidden = hidden[-2:, invert_idx]
    hidden = torch.cat((hidden[0], hidden[1]), -1)
    return hidden


def lstm_padded_custom(lstm, inputs, ls, num_layers):
    ls, idx = ls.sort(descending=True)
    invert_idx = idx.sort()[1]
    assert 0 <= idx.min() <= idx.max() < inputs.size(0), (inputs.size(0), idx.min(), idx.max())
    packed = pack_padded_sequence(inputs[idx], ls.tolist(), batch_first=True)
    _, (hidden, _) = lstm(packed)
    hidden = hidden[-num_layers*2:, invert_idx]
    hidden_ret = [hidden[0], hidden[1]]
    for i in range(1, num_layers):
        hidden_ret[0] += hidden[2*i]
        hidden_ret[1] += hidden[2*i+1]
    hidden = torch.cat([hidden_ret[0], hidden_ret[1]], -1)
    return hidden


class AbuseDetectionModel(nn.Module):
    def __init__(self, d_event, d_keycode, hidden_size, num_layers, dropout=0.5):
        super(AbuseDetectionModel, self).__init__()

        self.m_event_embeddings = nn.Embedding(3, d_event)
        self.ki_event_embeddings = nn.Embedding(2, d_event)
        self.kp_event_embeddings = nn.Embedding(2, d_event)
        self.keycode_embeddings = nn.Embedding(2257, d_keycode)

        self.m_zero = nn.Parameter(torch.zeros((1, d_event + 5), dtype=torch.float))
        self.ki_zero = nn.Parameter(torch.zeros((1, d_event + d_keycode + 1), dtype=torch.float))
        self.kp_zero = nn.Parameter(torch.zeros((1, d_event + 1), dtype=torch.float))

        self.input_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        #self.output_dropout = nn.Dropout(0.7)

        shared_lstm_args = {
            "hidden_size": hidden_size, "num_layers": num_layers,
            "batch_first": True, "dropout": dropout, "bidirectional": True,
        }
        self.lstm_m = LSTM(d_event + 5, **shared_lstm_args)
        self.lstm_ki = LSTM(d_event + d_keycode + 1, **shared_lstm_args)
        self.lstm_kp = LSTM(d_event + 1, **shared_lstm_args)

        self.fc = nn.Linear(6 * hidden_size, 2)

    def forward(self, inputs, lengths):
        m, ki, kp = inputs
        m_ls, ki_ls, kp_ls = lengths

        m_event_emb = self.m_event_embeddings(m["eventType"])
        m_ti = (m["timeInterval"].unsqueeze(-1) / 1000.).clamp(0, 10)
        m_absX = m["absX"].unsqueeze(-1) / 1200
        m_absY = m["absY"].unsqueeze(-1) / 600
        m_diffX = m["diffX"].unsqueeze(-1).clamp(-256, 256) / 32
        m_diffY = m["diffY"].unsqueeze(-1).clamp(-256, 256) / 32
        m_inputs = torch.cat((m_event_emb, m_ti, m_absX, m_absY, m_diffX, m_diffY), -1)
        m_zero = self.m_zero.unsqueeze(0).expand(m_inputs.size(0), -1, -1)
        m_inputs = torch.cat((m_inputs, m_zero), 1)
        m_inputs = self.input_dropout(m_inputs)
        m_hidden = lstm_padded(self.lstm_m, m_inputs, 1 + m_ls)

        ki_event_emb = self.ki_event_embeddings(ki["eventType"])
        keycode_emb = self.keycode_embeddings(ki["keyCode"])
        ki_ti = (ki["timeInterval"].unsqueeze(-1) / 1000.).clamp(0, 1)
        ki_inputs = torch.cat((ki_event_emb, keycode_emb, ki_ti), -1)
        ki_zero = self.ki_zero.unsqueeze(0).expand(ki_inputs.size(0), -1, -1)
        ki_inputs = torch.cat((ki_inputs, ki_zero), 1)
        ki_inputs = self.input_dropout(ki_inputs)
        ki_hidden = lstm_padded(self.lstm_ki, ki_inputs, 1 + ki_ls)

        kp_event_emb = self.kp_event_embeddings(kp["eventType"])
        kp_ti = (kp["timeInterval"].unsqueeze(-1) / 1000.).clamp(0, 1)
        kp_inputs = torch.cat((kp_event_emb, kp_ti), -1)
        kp_zero = self.kp_zero.unsqueeze(0).expand(kp_inputs.size(0), -1, -1)
        kp_inputs = torch.cat((kp_inputs, kp_zero), 1)
        kp_inputs = self.input_dropout(kp_inputs)
        kp_hidden = lstm_padded(self.lstm_kp, kp_inputs, 1 + kp_ls)

        output = torch.cat((m_hidden, ki_hidden, kp_hidden), -1)
        output = self.output_dropout(output)
        output = self.fc(output)
        return output


class AbuseDetectionModelV2(nn.Module):
    def __init__(self, d_event, d_keycode, hidden_size, num_layers, dropout=0.5):
        super(AbuseDetectionModelV2, self).__init__()

        self.m_event_embeddings = nn.Embedding(3, d_event)
        self.ki_event_embeddings = nn.Embedding(2, d_event)
        self.kp_event_embeddings = nn.Embedding(2, d_event)
        self.keycode_embeddings = nn.Embedding(2257, d_keycode)

        self.m_zero = nn.Parameter(torch.zeros((1, d_event + 5), dtype=torch.float))
        self.ki_zero = nn.Parameter(torch.zeros((1, d_event + d_keycode + 1), dtype=torch.float))
        self.kp_zero = nn.Parameter(torch.zeros((1, d_event + 1), dtype=torch.float))

        self.input_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        #self.output_dropout = nn.Dropout(0.7)

        shared_lstm_args = {
            "hidden_size": hidden_size, "num_layers": num_layers,
            "batch_first": True, "dropout": dropout, "bidirectional": True,
        }
        self.lstm_m = LSTM(d_event + 5, **shared_lstm_args)
        self.lstm_ki = LSTM(d_event + d_keycode + 1, **shared_lstm_args)
        self.lstm_kp = LSTM(d_event + 1, **shared_lstm_args)

        self.fc = nn.Sequential(
            self.output_dropout,
            nn.Linear(6 * hidden_size, 256),
            self.output_dropout,
            nn.Linear(256, 2)
        )

    def forward(self, inputs, lengths):
        m, ki, kp = inputs
        m_ls, ki_ls, kp_ls = lengths

        m_event_emb = self.m_event_embeddings(m["eventType"])
        m["timeInterval"] = (m["timeInterval"].unsqueeze(-1) / 1000.).clamp(0, 10)
        m["absX"] = m["absX"].unsqueeze(-1) / 1200
        m["absY"] = m["absY"].unsqueeze(-1) / 600
        m["diffX"] = m["diffX"].unsqueeze(-1).clamp(-256, 256) / 32
        m["diffY"] = m["diffY"].unsqueeze(-1).clamp(-256, 256) / 32
        m_inputs = torch.cat((m_event_emb, m["timeInterval"], m["absX"], m["absY"], m["diffX"], m["diffY"]), -1)
        m_zero = self.m_zero.unsqueeze(0).expand(m_inputs.size(0), -1, -1)
        m_inputs = torch.cat((m_inputs, m_zero), 1)
        m_inputs = self.input_dropout(m_inputs)
        m_hidden = lstm_padded(self.lstm_m, m_inputs, 1 + m_ls)

        ki_event_emb = self.ki_event_embeddings(ki["eventType"])
        keycode_emb = self.keycode_embeddings(ki["keyCode"])
        ki["timeInterval"] = (ki["timeInterval"].unsqueeze(-1) / 1000.).clamp(0, 1)
        ki_inputs = torch.cat((ki_event_emb, keycode_emb, ki["timeInterval"]), -1)
        ki_zero = self.ki_zero.unsqueeze(0).expand(ki_inputs.size(0), -1, -1)
        ki_inputs = torch.cat((ki_inputs, ki_zero), 1)
        ki_inputs = self.input_dropout(ki_inputs)
        ki_hidden = lstm_padded(self.lstm_ki, ki_inputs, 1 + ki_ls)

        kp_event_emb = self.kp_event_embeddings(kp["eventType"])
        kp["timeInterval"] = (kp["timeInterval"].unsqueeze(-1) / 1000.).clamp(0, 1)
        kp_inputs = torch.cat((kp_event_emb, kp["timeInterval"]), -1)
        kp_zero = self.kp_zero.unsqueeze(0).expand(kp_inputs.size(0), -1, -1)
        kp_inputs = torch.cat((kp_inputs, kp_zero), 1)
        kp_inputs = self.input_dropout(kp_inputs)
        kp_hidden = lstm_padded(self.lstm_kp, kp_inputs, 1 + kp_ls)

        output = torch.cat((m_hidden, ki_hidden, kp_hidden), -1)
        output = self.fc(output)
        return output


class AbuseDetectionModelV3(nn.Module):
    def __init__(self, d_event, d_keycode, hidden_size, num_layers, dropout=0.5, num_sum_layers=2):
        super(AbuseDetectionModelV3, self).__init__()

        self.m_event_embeddings = nn.Embedding(3, d_event)
        self.ki_event_embeddings = nn.Embedding(2, d_event)
        self.kp_event_embeddings = nn.Embedding(2, d_event)
        self.keycode_embeddings = nn.Embedding(2257, d_keycode)

        self.m_zero = nn.Parameter(torch.zeros((1, d_event + 5), dtype=torch.float))
        self.ki_zero = nn.Parameter(torch.zeros((1, d_event + d_keycode + 1), dtype=torch.float))
        self.kp_zero = nn.Parameter(torch.zeros((1, d_event + 1), dtype=torch.float))

        self.input_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        #self.output_dropout = nn.Dropout(0.7)

        shared_lstm_args = {
            "hidden_size": hidden_size, "num_layers": num_layers,
            "batch_first": True, "dropout": dropout, "bidirectional": True,
        }
        self.num_layers = shared_lstm_args['num_layers']
        self.lstm_m = LSTM(d_event + 5, **shared_lstm_args)
        self.lstm_ki = LSTM(d_event + d_keycode + 1, **shared_lstm_args)
        self.lstm_kp = LSTM(d_event + 1, **shared_lstm_args)

        self.num_sum_layers = num_sum_layers

        self.fc = nn.Linear(6 * hidden_size, 2)

    def forward(self, inputs, lengths):
        m, ki, kp = inputs
        m_ls, ki_ls, kp_ls = lengths

        m_event_emb = self.m_event_embeddings(m["eventType"])
        m_ti = (m["timeInterval"].unsqueeze(-1) / 1000.).clamp(0, 10)
        m_absX = m["absX"].unsqueeze(-1) / 1200
        m_absY = m["absY"].unsqueeze(-1) / 600
        m_diffX = m["diffX"].unsqueeze(-1).clamp(-256, 256) / 32
        m_diffY = m["diffY"].unsqueeze(-1).clamp(-256, 256) / 32
        m_inputs = torch.cat((m_event_emb, m_ti, m_absX, m_absY, m_diffX, m_diffY), -1)
        m_zero = self.m_zero.unsqueeze(0).expand(m_inputs.size(0), -1, -1)
        m_inputs = torch.cat((m_inputs, m_zero), 1)
        m_inputs = self.input_dropout(m_inputs)
        m_hidden = lstm_padded_custom(self.lstm_m, m_inputs, 1 + m_ls, self.num_sum_layers)

        ki_event_emb = self.ki_event_embeddings(ki["eventType"])
        keycode_emb = self.keycode_embeddings(ki["keyCode"])
        ki_ti = (ki["timeInterval"].unsqueeze(-1) / 1000.).clamp(0, 1)
        ki_inputs = torch.cat((ki_event_emb, keycode_emb, ki_ti), -1)
        ki_zero = self.ki_zero.unsqueeze(0).expand(ki_inputs.size(0), -1, -1)
        ki_inputs = torch.cat((ki_inputs, ki_zero), 1)
        ki_inputs = self.input_dropout(ki_inputs)
        ki_hidden = lstm_padded_custom(self.lstm_ki, ki_inputs, 1 + ki_ls, self.num_sum_layers)

        kp_event_emb = self.kp_event_embeddings(kp["eventType"])
        kp_ti = (kp["timeInterval"].unsqueeze(-1) / 1000.).clamp(0, 1)
        kp_inputs = torch.cat((kp_event_emb, kp_ti), -1)
        kp_zero = self.kp_zero.unsqueeze(0).expand(kp_inputs.size(0), -1, -1)
        kp_inputs = torch.cat((kp_inputs, kp_zero), 1)
        kp_inputs = self.input_dropout(kp_inputs)
        kp_hidden = lstm_padded_custom(self.lstm_kp, kp_inputs, 1 + kp_ls, self.num_sum_layers)

        output = torch.cat((m_hidden, ki_hidden, kp_hidden), -1)
        output = self.output_dropout(output)
        output = self.fc(output)
        return output


class GRUAbuseDetectionModel(nn.Module):
    def __init__(self, d_event, d_keycode, hidden_size, num_layers, dropout=0.5):
        super(GRUAbuseDetectionModel, self).__init__()

        self.m_event_embeddings = nn.Embedding(3, d_event)
        self.ki_event_embeddings = nn.Embedding(2, d_event)
        self.kp_event_embeddings = nn.Embedding(2, d_event)
        self.keycode_embeddings = nn.Embedding(2257, d_keycode)

        self.m_zero = nn.Parameter(torch.zeros((1, d_event + 5), dtype=torch.float))
        self.ki_zero = nn.Parameter(torch.zeros((1, d_event + d_keycode + 1), dtype=torch.float))
        self.kp_zero = nn.Parameter(torch.zeros((1, d_event + 1), dtype=torch.float))

        self.input_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        #self.output_dropout = nn.Dropout(0.7)

        shared_gru_args = {
            "hidden_size": hidden_size, "num_layers": num_layers,
            "batch_first": True, "dropout": dropout, "bidirectional": True,
        }
        self.gru_m = GRU(d_event + 5, **shared_gru_args)
        self.gru_ki = GRU(d_event + d_keycode + 1, **shared_gru_args)
        self.gru_kp = GRU(d_event + 1, **shared_gru_args)

        self.fc = nn.Linear(6 * hidden_size, 2)

    def forward(self, inputs, lengths):
        m, ki, kp = inputs
        m_ls, ki_ls, kp_ls = lengths

        m_event_emb = self.m_event_embeddings(m["eventType"])
        m_ti = (m["timeInterval"].unsqueeze(-1) / 1000.).clamp(0, 10)
        m_absX = m["absX"].unsqueeze(-1) / 1200
        m_absY = m["absY"].unsqueeze(-1) / 600
        m_diffX = m["diffX"].unsqueeze(-1).clamp(-256, 256) / 32
        m_diffY = m["diffY"].unsqueeze(-1).clamp(-256, 256) / 32
        m_inputs = torch.cat((m_event_emb, m_ti, m_absX, m_absY, m_diffX, m_diffY), -1)
        m_zero = self.m_zero.unsqueeze(0).expand(m_inputs.size(0), -1, -1)
        m_inputs = torch.cat((m_inputs, m_zero), 1)
        m_inputs = self.input_dropout(m_inputs)
        m_hidden = gru_padded(self.gru_m, m_inputs, 1 + m_ls)

        ki_event_emb = self.ki_event_embeddings(ki["eventType"])
        keycode_emb = self.keycode_embeddings(ki["keyCode"])
        ki_ti = (ki["timeInterval"].unsqueeze(-1) / 1000.).clamp(0, 1)
        ki_inputs = torch.cat((ki_event_emb, keycode_emb, ki_ti), -1)
        ki_zero = self.ki_zero.unsqueeze(0).expand(ki_inputs.size(0), -1, -1)
        ki_inputs = torch.cat((ki_inputs, ki_zero), 1)
        ki_inputs = self.input_dropout(ki_inputs)
        ki_hidden = gru_padded(self.gru_ki, ki_inputs, 1 + ki_ls)

        kp_event_emb = self.kp_event_embeddings(kp["eventType"])
        kp_ti = (kp["timeInterval"].unsqueeze(-1) / 1000.).clamp(0, 1)
        kp_inputs = torch.cat((kp_event_emb, kp_ti), -1)
        kp_zero = self.kp_zero.unsqueeze(0).expand(kp_inputs.size(0), -1, -1)
        kp_inputs = torch.cat((kp_inputs, kp_zero), 1)
        kp_inputs = self.input_dropout(kp_inputs)
        kp_hidden = gru_padded(self.gru_kp, kp_inputs, 1 + kp_ls)

        output = torch.cat((m_hidden, ki_hidden, kp_hidden), -1)
        output = self.output_dropout(output)
        output = self.fc(output)
        return output



class EnsembleModel(nn.Module):
    def __init__(
        self,
        m_emb_d_tf = 11, m_head_tf = 1, m_hid_tf = 64, m_layers_tf = 3,
        ki_emb_d_tf = 4, keycode_emb_d_tf = 11, ki_head_tf = 1, ki_hid_tf = 64, ki_layers_tf = 3,
        kp_emb_d_tf = 15, kp_head = 1, kp_hid_tf = 64, kp_layers = 3,
        dropout_tf = 0.5, classifier_dropout_tf = 0.5,
        d_event_sin_lstm = 4, d_keycode_sin_lstm = 4, hidden_size_sin_lstm = 16, num_layers_sin_lstm = 2, dropout_sin_lstm = 0.5,
        d_event_stck_lstm = 4, d_keycode_stck_lstm = 4, hidden_size_skct_lstm = 16, num_layers_stck_lstm = 2, dropout_stck_lstm = 0.5, num_sum_layers = 2,
        d_event_gru = 4, d_keycode_gru = 4, hidden_size_gru = 16, num_layers_gru = 2, dropout_gru = 0.5
    ):
        super(EnsembleModel, self).__init__()
        self.transformer_model = AbuseDetectorV3(
            m_emb_d=m_emb_d_tf, m_head=m_head_tf, m_hid=m_hid_tf, m_layers=m_layers_tf,
            ki_emb_d=ki_emb_d_tf, keycode_emb_d=keycode_emb_d_tf, ki_head=ki_head_tf, ki_hid=ki_hid_tf, ki_layers=ki_layers_tf,
            kp_emb_d=kp_emb_d_tf, kp_head=kp_head, kp_hid=kp_hid_tf, kp_layers=kp_layers,
            dropout=dropout_tf, classifier_dropout=classifier_dropout_tf
        )
        self.single_lstm_model = AbuseDetectionModel(
            d_event=d_event_sin_lstm, d_keycode=d_keycode_sin_lstm, hidden_size=hidden_size_sin_lstm,
            num_layers=num_layers_sin_lstm, dropout=dropout_sin_lstm
        )
        self.stacked_lstm_model = AbuseDetectionModelV3(
            d_event=d_event_stck_lstm, d_keycode=d_keycode_stck_lstm, hidden_size=hidden_size_skct_lstm,
            num_layers=num_layers_stck_lstm, dropout=dropout_stck_lstm,
            num_sum_layers=num_sum_layers
        )
        self.gru_model = GRUAbuseDetectionModel(
            d_event=d_event_gru, d_keycode=d_keycode_gru, hidden_size=hidden_size_gru,
            num_layers=num_layers_gru, dropout=dropout_gru
        )
    
    def forward(self, inputs, lengths, model_num=None):
        if model_num is None:
            inputs_tf = inputs
            inputs_sin_lstm = inputs
            inputs_stck_lstm = inputs
            inputs_gru = inputs
            output = self.transformer_model(inputs_tf)
            output += self.single_lstm_model(inputs_sin_lstm, lengths)
            output += self.stacked_lstm_model(inputs_stck_lstm, lengths)
            output += self.gru_model(inputs_gru, lengths)
            return output
        elif model_num==0:
            output = self.transformer_model(inputs)
            return output
        elif model_num==1:
            output = self.single_lstm_model(inputs, lengths)
            return output
        elif model_num==2:
            output = self.stacked_lstm_model(inputs, lengths)
            return output
        else:
            output = self.gru_model(inputs, lengths)
            return output


class EnsembleModelV2(nn.Module):
    def __init__(
        self,
        d_event_sin_lstm = 4, d_keycode_sin_lstm = 4, hidden_size_sin_lstm = 16, num_layers_sin_lstm = 2, dropout_sin_lstm = 0.5,
        d_event_stck_lstm = 4, d_keycode_stck_lstm = 4, hidden_size_skct_lstm = 16, num_layers_stck_lstm = 2, dropout_stck_lstm = 0.5, num_sum_layers = 2,
        d_event_gru = 4, d_keycode_gru = 4, hidden_size_gru = 16, num_layers_gru = 2, dropout_gru = 0.5
    ):
        super(EnsembleModelV2, self).__init__()
        self.single_lstm_model = AbuseDetectionModel(
            d_event=d_event_sin_lstm, d_keycode=d_keycode_sin_lstm, hidden_size=hidden_size_sin_lstm,
            num_layers=num_layers_sin_lstm, dropout=dropout_sin_lstm
        )
        self.stacked_lstm_model = AbuseDetectionModelV3(
            d_event=d_event_stck_lstm, d_keycode=d_keycode_stck_lstm, hidden_size=hidden_size_skct_lstm,
            num_layers=num_layers_stck_lstm, dropout=dropout_stck_lstm,
            num_sum_layers=num_sum_layers
        )
        self.gru_model = GRUAbuseDetectionModel(
            d_event=d_event_gru, d_keycode=d_keycode_gru, hidden_size=hidden_size_gru,
            num_layers=num_layers_gru, dropout=dropout_gru
        )
    
    def forward(self, inputs, lengths, model_num=None):
        if model_num is None:
            inputs_sin_lstm = inputs
            inputs_stck_lstm = inputs
            inputs_gru = inputs
            output = self.single_lstm_model(inputs_sin_lstm, lengths)
            output += self.stacked_lstm_model(inputs_stck_lstm, lengths)
            output += self.gru_model(inputs_gru, lengths)
            return output
        elif model_num==0:
            output = self.single_lstm_model(inputs, lengths)
            return output
        elif model_num==1:
            output = self.stacked_lstm_model(inputs, lengths)
            return output
        else:
            output = self.gru_model(inputs, lengths)
            return output


class EnsembleModelV3(nn.Module):
    def __init__(
        self,
        d_event_sin_lstm = 4, d_keycode_sin_lstm = 4, hidden_size_sin_lstm = 16, num_layers_sin_lstm = 2, dropout_sin_lstm = 0.5,
        d_event_stck_lstm = 4, d_keycode_stck_lstm = 4, hidden_size_skct_lstm = 16, num_layers_stck_lstm = 3, dropout_stck_lstm = 0.5, num_sum_layers = 2,
        d_event_gru = 4, d_keycode_gru = 4, hidden_size_gru = 16, num_layers_gru = 2, dropout_gru = 0.5
    ):
        super(EnsembleModelV3, self).__init__()
        self.single_lstm_model_01 = AbuseDetectionModel(
            d_event=d_event_sin_lstm, d_keycode=d_keycode_sin_lstm, hidden_size=hidden_size_sin_lstm,
            num_layers=num_layers_sin_lstm, dropout=0.5
        )
        self.single_lstm_model_02 = AbuseDetectionModel(
            d_event=d_event_sin_lstm, d_keycode=d_keycode_sin_lstm, hidden_size=hidden_size_sin_lstm*2,
            num_layers=num_layers_sin_lstm, dropout=0.7
        )
        self.stacked_lstm_model_01 = AbuseDetectionModelV3(
            d_event=d_event_stck_lstm, d_keycode=d_keycode_stck_lstm, hidden_size=hidden_size_skct_lstm,
            num_layers=num_layers_stck_lstm, dropout=0.5,
            num_sum_layers=num_sum_layers
        )
        self.stacked_lstm_model_02 = AbuseDetectionModelV3(
            d_event=d_event_stck_lstm, d_keycode=d_keycode_stck_lstm, hidden_size=hidden_size_skct_lstm*2,
            num_layers=num_layers_stck_lstm*2, dropout=0.7,
            num_sum_layers=num_sum_layers
        )
        self.gru_model_01 = GRUAbuseDetectionModel(
            d_event=d_event_gru, d_keycode=d_keycode_gru, hidden_size=hidden_size_gru,
            num_layers=num_layers_gru, dropout=0.5
        )
        self.gru_model_02 = GRUAbuseDetectionModel(
            d_event=d_event_gru, d_keycode=d_keycode_gru, hidden_size=hidden_size_gru*2,
            num_layers=num_layers_gru, dropout=0.7
        )
    
    def forward(self, inputs, lengths, model_num=None):
        if model_num is None:
            inputs_sin_lstm_01 = inputs
            inputs_sin_lstm_02 = inputs
            inputs_stck_lstm_01 = inputs
            inputs_stck_lstm_02 = inputs
            inputs_gru_01 = inputs
            inputs_gru_02 = inputs
            output = self.single_lstm_model_01(inputs_sin_lstm_01, lengths)
            output += self.single_lstm_model_02(inputs_sin_lstm_02, lengths)
            output += self.stacked_lstm_model_01(inputs_stck_lstm_01, lengths)
            output += self.stacked_lstm_model_02(inputs_stck_lstm_02, lengths)
            output += self.gru_model_01(inputs_gru_01, lengths)
            output += self.gru_model_02(inputs_gru_02, lengths)
            return output
        elif model_num==0:
            output = self.single_lstm_model_01(inputs, lengths)
            return output
        elif model_num==1:
            output = self.single_lstm_model_02(inputs, lengths)
            return output
        elif model_num==2:
            output = self.stacked_lstm_model_01(inputs, lengths)
            return output
        elif model_num==3:
            output = self.stacked_lstm_model_02(inputs, lengths)
            return output
        elif model_num==4:
            output = self.gru_model_01(inputs, lengths)
            return output
        else:
            output = self.gru_model_02(inputs, lengths)
            return output


class EnsembleModelV4(nn.Module):
    def __init__(
        self,
        d_event_sin_lstm = 4, d_keycode_sin_lstm = 4, hidden_size_sin_lstm = 16, num_layers_sin_lstm = 2,
    ):
        super(EnsembleModelV4, self).__init__()
        self.single_lstm_model_01 = AbuseDetectionModel(
            d_event=d_event_sin_lstm, d_keycode=d_keycode_sin_lstm, hidden_size=hidden_size_sin_lstm,
            num_layers=num_layers_sin_lstm, dropout=0.7
        )
        self.single_lstm_model_02 = AbuseDetectionModel(
            d_event=d_event_sin_lstm, d_keycode=d_keycode_sin_lstm, hidden_size=hidden_size_sin_lstm,
            num_layers=num_layers_sin_lstm, dropout=0.7
        )
        self.single_lstm_model_03 = AbuseDetectionModel(
            d_event=d_event_sin_lstm, d_keycode=d_keycode_sin_lstm, hidden_size=hidden_size_sin_lstm,
            num_layers=num_layers_sin_lstm, dropout=0.7
        )
        self.ensemble_w = nn.Parameter(torch.Tensor([1., 1., 1.]))
    
    def forward(self, inputs, lengths, model_num=None):
        if model_num is None:
            inputs_sin_lstm_01 = inputs
            inputs_sin_lstm_02 = inputs
            inputs_sin_lstm_03 = inputs
            output = self.single_lstm_model_01(inputs_sin_lstm_01, lengths)*self.ensemble_w[0]
            output += self.single_lstm_model_02(inputs_sin_lstm_02, lengths)*self.ensemble_w[1]
            output += self.single_lstm_model_03(inputs_sin_lstm_03, lengths)*self.ensemble_w[2]
            return output
        elif model_num==0:
            output = self.single_lstm_model_01(inputs, lengths)
            return output
        elif model_num==1:
            output = self.single_lstm_model_02(inputs, lengths)
            return output
        else:
            output = self.single_lstm_model_03(inputs, lengths)
            return output


class EnsembleModelV5(nn.Module):
    def __init__(
        self,
        d_event_sin_lstm = 4, d_keycode_sin_lstm = 4, hidden_size_sin_lstm = 16, num_layers_sin_lstm = 2,
    ):
        super(EnsembleModelV5, self).__init__()
        self.single_lstm_model_01 = AbuseDetectionModel(
            d_event=d_event_sin_lstm, d_keycode=d_keycode_sin_lstm, hidden_size=hidden_size_sin_lstm,
            num_layers=num_layers_sin_lstm, dropout=0.7
        )
        self.single_lstm_model_02 = AbuseDetectionModel(
            d_event=d_event_sin_lstm, d_keycode=d_keycode_sin_lstm, hidden_size=hidden_size_sin_lstm,
            num_layers=num_layers_sin_lstm, dropout=0.7
        )
        self.single_lstm_model_03 = AbuseDetectionModel(
            d_event=d_event_sin_lstm, d_keycode=d_keycode_sin_lstm, hidden_size=hidden_size_sin_lstm,
            num_layers=num_layers_sin_lstm, dropout=0.7
        )
        self.single_lstm_model_04 = AbuseDetectionModel(
            d_event=d_event_sin_lstm, d_keycode=d_keycode_sin_lstm, hidden_size=hidden_size_sin_lstm,
            num_layers=num_layers_sin_lstm, dropout=0.7
        )
        self.single_lstm_model_05 = AbuseDetectionModel(
            d_event=d_event_sin_lstm, d_keycode=d_keycode_sin_lstm, hidden_size=hidden_size_sin_lstm,
            num_layers=num_layers_sin_lstm, dropout=0.7
        )
        self.ensemble_w = nn.Parameter(torch.Tensor([1., 1.1, 1.2, 1.3, 1.4]))
    
    def forward(self, inputs, lengths, model_num=None):
        if model_num is None:
            output = self.single_lstm_model_01(inputs, lengths)*self.ensemble_w[0]
            output += self.single_lstm_model_02(inputs, lengths)*self.ensemble_w[1]
            output += self.single_lstm_model_03(inputs, lengths)*self.ensemble_w[2]
            output += self.single_lstm_model_04(inputs, lengths)*self.ensemble_w[3]
            output += self.single_lstm_model_05(inputs, lengths)*self.ensemble_w[4]
            return output
        elif model_num==0:
            output = self.single_lstm_model_01(inputs, lengths)
            return output
        elif model_num==1:
            output = self.single_lstm_model_02(inputs, lengths)
            return output
        else:
            output = self.single_lstm_model_03(inputs, lengths)
            return output

