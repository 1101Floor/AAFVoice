import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import multiprocessing
import dataset
import features_preprocessing
import myconfig
import torch.nn.functional as F


class BaseSpeakerEncoder(nn.Module):
    def _load_from(self, saved_model):
        var_dict = torch.load(saved_model, map_location=myconfig.DEVICE)
        self.load_state_dict(var_dict["encoder_state_dict"])


class Res2Conv1dReluBn(nn.Module):

    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    def __init__(self, channels, s=2):
        super().__init__()
        assert channels % s == 0, "{} % {} != 0".format(channels, s)
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
    return nn.Sequential(Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0), Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
                         Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0), SE_Connect(channels))


class FilterLayer(nn.Module):
    def __init__(self, nfilt=40, nfft=512):
        super(FilterLayer, self).__init__()
        self.nfilt = nfilt
        self.nfft = nfft
        # 从1到256生成随机的binpoints，确保数量为 nfilt + 2 并进行排序
        custom_binpoints = torch.randint(1, 257, (nfilt + 2,), dtype=torch.float32)
        # 对 binpoints 进行排序
        custom_binpoints = torch.sort(custom_binpoints).values
        self.binpoint_params = nn.Parameter(custom_binpoints)

    def forward(self, x):
        binpoints = torch.sort(self.binpoint_params).values
        # add func
        fbank = torch.zeros((self.nfilt, self.nfft // 2 + 1), device=x.device)  # Adjusted to match x_rest size
        # 使用三角滤波器生成
        for j in range(self.nfilt - 1):  # 确保不超出 binpoints 索引范围
            # 第一部分：上升沿
            for i in range(int(binpoints[j]), int(binpoints[j + 1])):
                fbank[j, i] = (i - binpoints[j]) / ((binpoints[j + 1] - binpoints[j]) ** 2)
            # 第二部分：下降沿
            if j + 2 < len(binpoints):  # 确保 binpoints[j + 2] 存在
                for i in range(int(binpoints[j + 1]), int(binpoints[j + 2])):
                    fbank[j, i] = (binpoints[j + 2] - i) / ((binpoints[j + 2] - binpoints[j + 1]) ** 2)
        first_column = x[:, :, 0:1]
        filtered = torch.matmul(x, fbank.t())  # Matrix multiplication
        filtered[:, :, 0:1] = first_column
        return filtered


class LstmSpeakerEncoder(BaseSpeakerEncoder):
    def __init__(self, saved_model=""):
        super(LstmSpeakerEncoder, self).__init__()
        # Define the FilterLayer
        self.filter_encoder = FilterLayer(nfilt=40, nfft=512)
        # Define the LSTM network.
        self.lstm = nn.LSTM(input_size=myconfig.N_MFCC, hidden_size=myconfig.LSTM_HIDDEN_SIZE, num_layers=myconfig.LSTM_NUM_LAYERS, batch_first=True, bidirectional=myconfig.BI_LSTM)
        # Load from a saved model if provided.
        if saved_model:
            self._load_from(saved_model)

    def _load_from(self, saved_model):
        checkpoint = torch.load(saved_model, map_location=myconfig.DEVICE)
        self.load_state_dict(checkpoint['encoder_state_dict'])
        self.filter_encoder.binpoint_params.data = checkpoint['binpoint_params']

    def _aggregate_frames(self, batch_output):
        """Aggregate output frames."""
        if myconfig.FRAME_AGGREGATION_MEAN:
            return torch.mean(batch_output, dim=1, keepdim=False)
        else:
            return batch_output[:, -1, :]

    def remove_zero_rows(self, x):
        """Remove rows that are all zeros across the feature dimension."""
        # Sum across the feature dimension (dim=2), resulting in a shape of [batch_size, sequence_length]
        non_zero_indices = (x.sum(dim=2) != 0)
        # Iterate over each batch and filter out zero rows
        filtered_x = [sample[mask] for sample, mask in zip(x, non_zero_indices)]
        # Pad sequences to make sure they have the same length (if necessary)
        filtered_x = nn.utils.rnn.pad_sequence(filtered_x, batch_first=True)
        return filtered_x

    def forward(self, x):
        # Pass through FilterLayer
        x = self.filter_encoder(x)
        # New layer: Remove rows that are all zeros
        x = self.remove_zero_rows(x)
        x = torch.log(x + 1e-10)
        # Prepare LSTM initial hidden and cell states
        D = 2 if myconfig.BI_LSTM else 1
        h0 = torch.zeros(D * myconfig.LSTM_NUM_LAYERS, x.shape[0], myconfig.LSTM_HIDDEN_SIZE).to(myconfig.DEVICE)
        c0 = torch.zeros(D * myconfig.LSTM_NUM_LAYERS, x.shape[0], myconfig.LSTM_HIDDEN_SIZE).to(myconfig.DEVICE)
        # Pass through LSTM
        y, (hn, cn) = self.lstm(x, (h0, c0))
        # Aggregate frames
        print("正在使用LSTM")
        return self._aggregate_frames(y)


class GRUSpeakerEncoder(BaseSpeakerEncoder):
    def __init__(self, saved_model=""):
        super(GRUSpeakerEncoder, self).__init__()
        # Define the FilterLayer
        self.filter_encoder = FilterLayer(nfilt=40, nfft=512)
        # Define the GRU network.
        self.GRU = nn.GRU(input_size=myconfig.N_MFCC, hidden_size=myconfig.GRU_HIDDEN_SIZE, num_layers=myconfig.GRU_NUM_LAYERS, batch_first=True, bidirectional=myconfig.BI_GRU)
        # Load from a saved model if provided.
        if saved_model:
            self._load_from(saved_model)

    def _load_from(self, saved_model):
        checkpoint = torch.load(saved_model, map_location=myconfig.DEVICE)
        self.load_state_dict(checkpoint['encoder_state_dict'])
        self.filter_encoder.binpoint_params.data = checkpoint['binpoint_params']

    def _aggregate_frames(self, batch_output):
        """Aggregate output frames."""
        if myconfig.FRAME_AGGREGATION_MEAN:
            return torch.mean(batch_output, dim=1, keepdim=False)
        else:
            return batch_output[:, -1, :]

    def remove_zero_rows(self, x):
        """Remove rows that are all zeros across the feature dimension."""
        # Sum across the feature dimension (dim=2), resulting in a shape of [batch_size, sequence_length]
        non_zero_indices = (x.sum(dim=2) != 0)
        # Iterate over each batch and filter out zero rows
        filtered_x = [sample[mask] for sample, mask in zip(x, non_zero_indices)]
        # Pad sequences to make sure they have the same length (if necessary)
        filtered_x = nn.utils.rnn.pad_sequence(filtered_x, batch_first=True)
        return filtered_x

    def forward(self, x):
        # Pass through FilterLayer
        x = self.filter_encoder(x)
        # New layer: Remove rows that are all zeros
        x = self.remove_zero_rows(x)
        x = torch.log(x + 1e-10)
        # Prepare LSTM initial hidden and cell states
        D = 2 if myconfig.BI_GRU else 1
        h0 = torch.zeros(D * myconfig.GRU_NUM_LAYERS, x.shape[0], myconfig.GRU_HIDDEN_SIZE).to(myconfig.DEVICE)
        y, hn = self.GRU(x, h0)
        print("正在使用GRU")
        return self._aggregate_frames(y)


class TransformerSpeakerEncoder(BaseSpeakerEncoder):
    def __init__(self, saved_model=""):
        super(TransformerSpeakerEncoder, self).__init__()
        # Define the FilterLayer
        self.filter_encoder = FilterLayer(nfilt=80, nfft=512)

        # Define the Transformer network
        self.linear_layer = nn.Linear(myconfig.N_MFCC, myconfig.TRANSFORMER_DIM)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=myconfig.TRANSFORMER_DIM, nhead=myconfig.TRANSFORMER_HEADS, batch_first=True),
                                             num_layers=myconfig.TRANSFORMER_ENCODER_LAYERS)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=myconfig.TRANSFORMER_DIM, nhead=myconfig.TRANSFORMER_HEADS, batch_first=True), num_layers=1)

        # Load from a saved model if provided
        if saved_model:
            self._load_from(saved_model)

    def _load_from(self, saved_model):
        checkpoint = torch.load(saved_model, map_location=myconfig.DEVICE)
        self.load_state_dict(checkpoint['encoder_state_dict'])
        self.filter_encoder.binpoint_params.data = checkpoint['binpoint_params']

    def remove_zero_rows(self, x):
        """Remove rows that are all zeros across the feature dimension."""
        non_zero_indices = (x.sum(dim=2) != 0)
        filtered_x = [sample[mask] for sample, mask in zip(x, non_zero_indices)]
        filtered_x = nn.utils.rnn.pad_sequence(filtered_x, batch_first=True)
        return filtered_x

    def forward(self, x):
        # Pass through FilterLayer
        x = self.filter_encoder(x)
        # Remove rows that are all zeros
        x = self.remove_zero_rows(x)
        # Pass through Transformer
        encoder_input = torch.sigmoid(self.linear_layer(x))
        encoder_output = self.encoder(encoder_input)
        tgt = torch.zeros(x.shape[0], 1, myconfig.TRANSFORMER_DIM).to(myconfig.DEVICE)
        output = self.decoder(tgt, encoder_output)
        print("正在使用Transformer训练binpoint")
        return output[:, 0, :]


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(input_dim, input_dim)  # 用于生成注意力权重
        self.softmax = nn.Softmax(dim=-1)  # 在特征维度上归一化权重

    def forward(self, x):
        # 输入维度: [batch, dim]
        weights = self.softmax(self.attention_weights(x))  # 注意力权重: [batch, dim]
        output = weights * x  # 加权后的输出: [batch, dim]
        return output


class ResCNN_ASP_SpeakerEncoder(BaseSpeakerEncoder):
    def __init__(self, saved_model=""):
        super(ResCNN_ASP_SpeakerEncoder, self).__init__()
        # Define the FilterLayer
        self.filter_encoder = FilterLayer(nfilt=80, nfft=512)
        # Define the TDNN network.
        self.Linear1 = nn.Linear(80, 64)
        self.conv1 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.pooling = AttentiveStatsPool(128, 64)
        self.bn1 = nn.GroupNorm(num_groups=1, num_channels=256)
        # self.attention_layer = Attention(input_dim=256)
        self.linear2 = nn.Linear(256, 512)
        # Load from a saved model if provided
        if saved_model:
            self._load_from(saved_model)

    def _load_from(self, saved_model):
        checkpoint = torch.load(saved_model, map_location=myconfig.DEVICE)
        self.load_state_dict(checkpoint['encoder_state_dict'])
        self.filter_encoder.binpoint_params.data = checkpoint['binpoint_params']

    def remove_zero_rows(self, x):
        """Remove rows that are all zeros across the feature dimension."""
        non_zero_indices = (x.sum(dim=2) != 0)
        filtered_x = [sample[mask] for sample, mask in zip(x, non_zero_indices)]
        filtered_x = nn.utils.rnn.pad_sequence(filtered_x, batch_first=True)
        return filtered_x

    def forward(self, x):
        x = self.filter_encoder(x)
        # add new layer
        # Remove rows that are all zeros
        x = self.remove_zero_rows(x)
        x = torch.relu(self.Linear1(x))
        tranx = x.transpose(1, 2)
        input1 = self.conv1(tranx)
        input2 = self.conv2(input1) + input1
        input3 = self.conv3(input1 + input2) + input1 + input2
        out = self.pooling(input3)
        # print(out.shape)
        # out = self.attention_layer(out)  # 注意力操作，仍为 [batch, 256]
        out = self.bn1(out)
        out = self.linear2(out)
        print('正在使用ResCNN')
        return out


# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#
# model = TransformerSpeakerEncoder()
# print(f"Total parameters in model: {count_parameters(model)}")
# if __name__ == '__main__':
#     # Input size: batch_size * seq_len * feat_dim
#     x = torch.rand(24, 100, 257)
#     x = x.to(myconfig.DEVICE)
#     model = TDNNSpeakerEncoder()
#     model.to(myconfig.DEVICE)
#     out = model(x)

'''
每一次训练的时候检查一下使用模型：
'''


def get_speaker_encoder(load_from=""):
    """Create speaker encoder model or load it from a saved model."""
    if myconfig.USE_TRANSFORMER:
        # model = TransformerSpeakerEncoder()
        model = ResCNN_ASP_SpeakerEncoder()
    else:
        # model = LstmSpeakerEncoder()
        model = GRUSpeakerEncoder()
    if load_from:
        model._load_from(load_from)
    return model.to(myconfig.DEVICE)


def get_triplet_loss(anchor, pos, neg):
    """Triplet loss defined in https://arxiv.org/pdf/1705.02304.pdf."""
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    return torch.maximum(cos(anchor, neg) - cos(anchor, pos) + myconfig.TRIPLET_ALPHA, torch.tensor(0.0))


def get_triplet_loss_from_batch_output(batch_output, batch_size):
    """Triplet loss from N*(a|p|n) batch output."""
    batch_output_reshaped = torch.reshape(batch_output, (batch_size, 3, batch_output.shape[1]))
    batch_loss = get_triplet_loss(batch_output_reshaped[:, 0, :], batch_output_reshaped[:, 1, :], batch_output_reshaped[:, 2, :])
    loss = torch.mean(batch_loss)
    return loss


def save_model(saved_model_path, encoder, losses, start_time):
    """Save model to disk."""
    training_time = time.time() - start_time
    os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
    if not saved_model_path.endswith(".pt"):
        saved_model_path += ".pt"
    torch.save({"encoder_state_dict": encoder.state_dict(), "losses": losses, "training_time": training_time, "binpoint_params": encoder.filter_encoder.binpoint_params.data}, saved_model_path)


def train_network(spk_to_utts, num_steps, saved_model=None, pool=None):
    start_time = time.time()
    losses = []
    encoder = get_speaker_encoder()
    # Train
    binpoint_params = encoder.filter_encoder.binpoint_params
    other_params = [p for n, p in encoder.named_parameters() if p is not binpoint_params]
    # optimizer = optim.Adam(encoder.parameters(), lr=myconfig.LEARNING_RATE)
    optimizer = optim.Adam([{'params': binpoint_params, 'lr': 0.005},  # Set custom learning rate for binpoint_params
                            {'params': other_params, 'lr': myconfig.LEARNING_RATE}  # Set standard learning rate for all other parameters
                            ])

    print("Start training")
    for step in range(num_steps):
        optimizer.zero_grad()

        # Build batched input.
        batch_input = features_preprocessing.get_batched_triplet_input(spk_to_utts, myconfig.BATCH_SIZE, pool).to(myconfig.DEVICE)

        # Compute loss.
        batch_output = encoder(batch_input)
        loss = get_triplet_loss_from_batch_output(batch_output, myconfig.BATCH_SIZE)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print("step:", step, "/", num_steps, "loss:", loss.item())

        if (saved_model is not None and (step + 1) % myconfig.SAVE_MODEL_FREQUENCY == 0):
            checkpoint = saved_model
            if checkpoint.endswith(".pt"):
                checkpoint = checkpoint[:-3]
            checkpoint += ".ckpt-" + str(step + 1) + ".pt"
            save_model(checkpoint, encoder, losses, start_time)

    training_time = time.time() - start_time
    print("Finished training in", training_time, "seconds")
    if saved_model is not None:
        save_model(saved_model, encoder, losses, start_time)
    return losses


def continual_train_network(spk_to_utts, num_steps, saved_model=None, pool=None):
    start_time = time.time()
    losses = []
    # Load the pre-trained model
    encoder = get_speaker_encoder(load_from=saved_model)
    # Freeze specific layers
    for param in encoder.filter_encoder.parameters():
        param.requires_grad = False
    # Freeze log transformation part by setting requires_grad to False on the input itself if necessary
    # But since torch.log doesn't have parameters, we just don't need to involve it in optimization
    # Only pass the parameters you want to train to the optimizer
    optimizer = optim.Adam([{'params': [p for p in encoder.parameters() if p.requires_grad], 'lr': myconfig.LEARNING_RATE}])
    print("Start continual training")
    for step in range(num_steps):
        optimizer.zero_grad()
        # Build batched input.
        batch_input = features_preprocessing.get_batched_triplet_input(spk_to_utts, myconfig.BATCH_SIZE, pool).to(myconfig.DEVICE)
        # Compute loss.
        batch_output = encoder(batch_input)
        loss = get_triplet_loss_from_batch_output(batch_output, myconfig.BATCH_SIZE)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print("step:", step, "/", num_steps, "loss:", loss.item())

        if (saved_model is not None and (step + 1) % myconfig.SAVE_MODEL_FREQUENCY == 0):
            checkpoint = saved_model
            if checkpoint.endswith(".pt"):
                checkpoint = checkpoint[:-3]
            checkpoint += ".ckpt-" + str(step + 1) + ".pt"
            save_model(checkpoint, encoder, losses, start_time)

    training_time = time.time() - start_time
    print("Finished continual training in", training_time, "seconds")
    if saved_model is not None:
        save_model(saved_model, encoder, losses, start_time)
    return losses


def run_training():
    if myconfig.TRAIN_DATA_CSV:
        spk_to_utts = dataset.get_csv_spk_to_utts(myconfig.TRAIN_DATA_CSV)
        print("Training data:", myconfig.TRAIN_DATA_CSV)
    else:
        spk_to_utts = dataset.get_librispeech_spk_to_utts(myconfig.TRAIN_DATA_DIR)
        print("Training data:", myconfig.TRAIN_DATA_DIR)
    with multiprocessing.Pool(myconfig.NUM_PROCESSES) as pool:
        losses = train_network(spk_to_utts, myconfig.TRAINING_STEPS, myconfig.SAVED_BINPOINT_PATH, pool)
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()


def continual_run_training():
    if myconfig.TRAIN_DATA_CSV:
        spk_to_utts = dataset.get_csv_spk_to_utts(myconfig.TRAIN_DATA_CSV)
        print("Training data:", myconfig.TRAIN_DATA_CSV)
    else:
        spk_to_utts = dataset.get_librispeech_spk_to_utts(myconfig.TRAIN_DATA_DIR)
        print("Training data:", myconfig.TRAIN_DATA_DIR)
    with multiprocessing.Pool(myconfig.NUM_PROCESSES) as pool:
        losses = continual_train_network(spk_to_utts, myconfig.TRAINING_STEPS, myconfig.SAVED_BINPOINT_PATH, pool)
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()


if __name__ == "__main__":
    continual_run_training()
    # run_training()
