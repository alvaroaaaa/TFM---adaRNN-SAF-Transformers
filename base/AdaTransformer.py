import torch
import torch.nn as nn
import math

from base.loss_transfer import TransferLoss
import torch.nn.functional as F


class AdaTransformer(nn.Module):

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 h: int,
                 N: int,
                 dropout: float = 0.3,
                 pe_period: int = 24):
        super().__init__()

        self._d_model = d_model
        self._pe_period = pe_period
        self.num_layers = N

        self.encoder_stacks = []
        self.decoder_stacks = []
        self.backcast_decoder_stacks = []

        for _ in range(N):
            # Create a TransformerEncoderLayer and TransformerEncoder for each stack
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=h,
                dim_feedforward=d_model,
                dropout=dropout,
                batch_first=True
            )
            encoder_stack = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.encoder_stacks.append(encoder_stack)

            # Create a TransformerDecoderLayer and TransformerDecoder for each stack
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=h,
                dim_feedforward=d_model,
                dropout=dropout,
                batch_first=True
            )
            decoder_stack = nn.TransformerDecoder(decoder_layer, num_layers=1)
            self.decoder_stacks.append(decoder_stack)

            # Create an additional TransformerDecoder for backcasting, if needed
            backcast_decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=h,
                dim_feedforward=d_model,
                dropout=dropout,
                batch_first=True
            )
            backcast_decoder_stack = nn.TransformerDecoder(backcast_decoder_layer, num_layers=1)
            self.backcast_decoder_stacks.append(backcast_decoder_stack)
        
        self.layers_encoding = nn.ModuleList(self.encoder_stacks)
        self.layers_decoding = nn.ModuleList(self.decoder_stacks)
        self.layers_backasting = nn.ModuleList(self.backcast_decoder_stacks)

        self._embedding = nn.Linear(d_input, d_model)

        self.bottleneck = nn.Sequential(
            nn.Linear(d_model, pe_period),
            nn.ReLU(),
            nn.Linear(pe_period, pe_period),
            nn.BatchNorm1d(pe_period),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(pe_period, d_output)

        self.gate = nn.ModuleList()
        self.bn_lst = nn.ModuleList()
        for _ in range(N):
            # Assuming d_model is used as the feature dimension for each Transformer layer
            gate_weight = nn.Linear(pe_period * d_model * 2, pe_period)
            self.gate.append(gate_weight)
            
            bn_layer = nn.BatchNorm1d(pe_period)
            self.bn_lst.append(bn_layer)
        
        self.softmax = torch.nn.Softmax(dim=0)
        self.init_layers()


    def init_layers(self):
        for gate_weight in self.gate:
            nn.init.normal_(gate_weight.weight, mean=0, std=0.05)
            nn.init.constant_(gate_weight.bias, 0)

    def generate_positional_encodings(self, len_seq, n_input):
        position = torch.arange(len_seq).unsqueeze(1).float()
        # Generate the frequencies for the positional encodings
        div_term = torch.exp(torch.arange(0, n_input, 2).float() * -(math.log(10000.0) / n_input))

        # Initialize the positional encoding matrix
        positional_encoding = torch.zeros(len_seq, n_input)

        # Apply sine to even indices in the positional encoding matrix
        positional_encoding[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices in the positional encoding matrix
        # The adjustment for odd indices div_term is no longer needed due to proper alignment
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        return positional_encoding

    def forward_Boosting(self, x, weight_mat=None):
        # Process input through Transformer and apply gating mechanism
        fc_out, out_list, out_weight_list = self.transformer_features(x)
        
        # Split the Transformer output into source and target features
        out_list_s, out_list_t = self.get_features(out_list)
        
        # Initialize transfer loss and distance matrix
        loss_transfer = torch.zeros((1,)).cuda()
        dist_mat = torch.zeros(self.num_layers, self._pe_period).cuda()  # Adjust dimensions if necessary
        
        # Initialize or use provided weight matrix for boosting
        if weight_mat is None:
            weight = (1.0 / self._pe_period * torch.ones(self.num_layers, self._pe_period)).cuda()
        else:
            weight = weight_mat
        
        # Calculate transfer loss for each pair of source and target features
        # For each layer in the stack
        for i in range(len(out_list_s)):
            criterion_transfer = TransferLoss(loss_type='cosine', input_dim=out_list_s[i].shape[2])
            for j in range(self._pe_period):
                # Compute transfer loss between source and target features at each sequence position
                loss_trans = criterion_transfer.compute(out_list_s[i][:, j, :], out_list_t[i][:, j, :])
                loss_transfer += weight[i, j] * loss_trans
                dist_mat[i, j] = loss_trans
        
        return fc_out, loss_transfer, dist_mat, weight

    def encode_and_backcast(self, x, mask_ratio=0.3):
        x = self._embedding(x)
        seq_pe = self.generate_positional_encodings(self._pe_period, self._d_model).to(x.device)
        out = x + seq_pe

        for layer in self.layers_encoding:
            out = layer(out)

        seq_pe = self.generate_positional_encodings(self._pe_period, self._d_model).to(out.device)

        out = out + seq_pe

        # Create a mask
        mask = torch.rand(out.shape, device=out.device) < mask_ratio
        # Apply mask
        masked_out = out.masked_fill(mask, 0)

        decoder_input = masked_out

        # Backcasting using the masked output
        for layer in self.layers_backasting:
            decoder_input = layer(decoder_input, masked_out)

        # Calculate loss (MSE between backcast and original input before projection)
        backcast_loss = F.mse_loss(decoder_input, x)

        return backcast_loss

    def apply_gating(self, transformer_output, index):
        # Split output for source and target (assuming domain adaptation scenario)
        x_s = transformer_output[:transformer_output.size(0) // 2]
        x_t = transformer_output[transformer_output.size(0) // 2:]

        # Concatenate source and target outputs for gating
        x_all = torch.cat((x_s, x_t), dim=2)  # Adjust dimensions as needed
        x_all = x_all.view(x_all.size(0), -1) # shape instead of size?
        

        # Apply gating mechanism for each layer (if multiple layers are used)
        weight = torch.sigmoid(self.bn_lst[index](self.gate[index](x_all.float())))
        weight = torch.mean(weight, dim=0)
        
        return self.softmax(weight).squeeze()

    def transformer_features(self, x):
        K = x.shape[1]

        # Embeddin module
        encoding = self._embedding(x)

        seq_pe = self.generate_positional_encodings(K, self._d_model).to(x.device)

        encoding = encoding + seq_pe

        # Encoding stack
        list_encoding = []
        out_weight_list = []
        for i, layer in enumerate(self.layers_encoding):
            encoding = layer(encoding)
            list_encoding.append(encoding)
            out_weight_list.append(self.apply_gating(encoding, i))

        # Decoding stack
        decoding = encoding

        seq_pe = self.generate_positional_encodings(K, self._d_model).to(x.device)
        decoding = decoding + seq_pe

        for layer in self.layers_decoding:
            decoding = layer(decoding, encoding)

        # Output module
        output = self.bottleneck(decoding)
        output = self.fc(output)
        output = torch.sigmoid(output)
        return output, list_encoding, out_weight_list


    def forward_pre_train(self, x, loss_type, len_win=0):
        out, out_list_all, out_weight_list = self.transformer_features(x)

        out_list_s, out_list_t = self.get_features(out_list_all)

        loss_transfer = torch.zeros((1,)).cuda()

        for i in range(len(out_list_s)):
            criterion_transfer = TransferLoss(loss_type=loss_type, input_dim=out_list_s[i].shape[2])
            for j in range(self._pe_period):
                # Adjust window for calculating transfer loss
                i_start = max(j - len_win, 0)
                i_end = min(j + len_win, self._pe_period - 1)
                for k in range(i_start, i_end + 1):
                    weight = out_weight_list[i][j]
                    # Compute transfer loss between source and target features within the window
                    loss_transfer += weight * criterion_transfer.compute(out_list_s[i][:, j, :], out_list_t[i][:, k, :])

        return out, loss_transfer, out_weight_list

    def adapt_encoding_weight(self, list_encoding, loss_type, train_type, weight_mat=None):
        loss_all = torch.zeros(1).cuda()
        len_seq = list_encoding[0].shape[1]
        num_layers = len(list_encoding)
        if weight_mat is None:
            weight = (1.0 / len_seq *
                      torch.ones(num_layers, len_seq)).cuda()
        else:
            weight = weight_mat
        dist_mat = torch.zeros(num_layers, len_seq).cuda()

        out_list_s, out_list_t = self.get_features(list_encoding)

        for i in range(len(list_encoding)):
            criterion_transder = TransferLoss(
                loss_type=loss_type, input_dim=out_list_s[i].shape[2])
            for j in range(out_list_s[i].shape[1]):
                loss_transfer = criterion_transder.compute(out_list_s[i][:, j, :], out_list_t[i][:, j, :])
                loss_all = loss_all + weight[i, j] * loss_transfer
                dist_mat[i, j] = loss_transfer 
        
        return loss_all, dist_mat, weight

    def get_features(self, output_list):
        fea_list_src, fea_list_tar = [], []
        for output in output_list:
            fea_list_src.append(output[:output.size(0) // 2])
            fea_list_tar.append(output[output.size(0) // 2:])
        return fea_list_src, fea_list_tar

    def update_weight_Boosting(self, weight_mat, dist_old, dist_new):
        epsilon = 1e-12
        dist_old = dist_old.detach()
        dist_new = dist_new.detach()
        ind = dist_new > dist_old + epsilon
        weight_mat[ind] = weight_mat[ind] * \
            (1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))
        weight_norm = torch.norm(weight_mat, dim=1, p=1)
        weight_mat = weight_mat / weight_norm.t().unsqueeze(1).repeat(1, len(weight_mat[0]))
        return weight_mat

    def predict(self, x):
        # Encoder output
        x = self._embedding(x)

        seq_pe = self.generate_positional_encodings(self._pe_period, self._d_model).to(x.device)
        out = x + seq_pe

        for layer in self.layers_encoding:
            out = layer(out)
        
        seq_pe = self.generate_positional_encodings(self._pe_period, self._d_model).to(out.device)

        out = out + seq_pe

        decoder_input = out
   
        # Apply decoder to generate the next step
        for layer in self.layers_decoding:
            decoder_input = layer(decoder_input, out)
        

        prediction = torch.sigmoid(self.fc(self.bottleneck(decoder_input[:, -1, :])))

     
        return prediction