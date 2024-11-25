import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
import torch.nn.functional as F

class GRUConvTransformerModel(nn.Module):
    def __init__(
            self, 
            input_dim, 
            gru_hidden_dim, 
            transformer_d_model, 
            num_heads, 
            num_decoder_layers,
            vocab_size
        ):
        super(GRUConvTransformerModel, self).__init__()

        self.gru_hidden_dim = gru_hidden_dim
        self.transformer_d_model = transformer_d_model

        # BERT Encoder
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.bert_encoder.resize_token_embeddings(vocab_size)

        # GRU layer
        self.gru = nn.GRU(input_dim, gru_hidden_dim, batch_first=True) # (batch size, seq_len, hidden_size))

        # Transformer Decoder
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=transformer_d_model, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_decoder_layers)

        # # Linear layer for gru to decoder
        # self.linear = nn.Linear(self.gru_hidden_dim, self.transformer_d_model)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.2)

        # Max pooling
        self.max_pooling = nn.MaxPool1d(kernel_size=2, stride=2)

        # Linear layer for output
        self.fc_out = nn.Linear(transformer_d_model, self.bert_encoder.config.vocab_size)

        # Positional Encoding for the target sequence
        self.positional_encoding = nn.Embedding(1024, transformer_d_model)

        # Embedding layer for target tokens
        self.tgt_embedding = nn.Embedding(self.bert_encoder.config.vocab_size, transformer_d_model)

    def forward(self, tgt, x):
        """
        Forward pass through the model.

        Args:
            x: Input sequence, shape: (batch_size, n, input_dim)
            tgt: Target sequence for transformer decoder, shape: (batch_size, n_tgt)

        Returns:
            Output of the model.
        """
        # Target output embeddings
        tgt_embedding = self.tgt_embedding(tgt)
        tgt_embedding += self.positional_encoding(torch.arange(tgt.size(1), device=x.device)).unsqueeze(0)
        tgt_embedding = tgt_embedding.transpose(0, 1)

        # GRU layer
        gru_out, _ = self.gru(x) # (batch_size, seq_len, hidden_dim)

        # Apply MaxPool1d to GRU output
        gru_out = gru_out.transpose(1, 2)  # Change shape to (batch_size, hidden_dim, seq_len)
        gru_out = self.max_pooling(gru_out)  # Apply MaxPooling (batch_size, hidden_dim, pooled_seq_len)

        # After pooling, reshape back to (batch_size, pooled_seq_len, hidden_dim)
        gru_out = gru_out.transpose(1, 2)

        # linear layer
        memory = self.dropout(gru_out) # (batch_size, seq_len, d_model)

        # Prepare memory for the Transformer decoder
        memory = memory.permute(1, 0, 2)
        transformer_out = self.transformer_decoder(tgt_embedding, memory) # Output shape: (seq_len_tgt, batch_size, d_model)

        # Apply linear layer for final output
        outputs = self.fc_out(transformer_out).permute(1, 0, 2)
        return outputs
    

    