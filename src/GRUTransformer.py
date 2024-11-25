import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import GRUConvTransformerModel


class GRUConvTransformer:
    def __init__(self, name='unknown', input_dim=768, gru_hidden_dim=256, kernel_size=3, 
                 transformer_d_model=512, num_heads=8, num_decoder_layers=12, learning_rate=5e-5, vocab_size=30524, device=None):
        """
        Initializes the GRUConvTransformer model and training utilities.

        Args:
            input_dim: Input dimensionality of BERT-encoded features.
            gru_hidden_dim: Hidden dimensionality for the GRU layer.
            transformer_d_model: Dimensionality of the Transformer decoder model.
            num_heads: Number of attention heads in the Transformer decoder.
            num_decoder_layers: Number of Transformer decoder layers.
            learning_rate: Learning rate for the optimizer.
            device: Device for computation ('cuda' or 'cpu').
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print("Using cuda device.")
        else:
            print("Using cpu device.")

        self.name = name

        # Initialize model components
        self.model = GRUConvTransformerModel(
            input_dim=input_dim,
            gru_hidden_dim=gru_hidden_dim,
            transformer_d_model=transformer_d_model,
            num_heads=num_heads,
            num_decoder_layers=num_decoder_layers,
            vocab_size=vocab_size
        ).to(self.device)

        # Define optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def up_data_lr(self,learning_rate):
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, train_loader, val_loader, epochs=10):
        """
        Trains the model on the given dataset.

        Args:
            train_loader: DataLoader for the training data.
            val_loader: DataLoader for the validation data.
            epochs: Number of training epochs.
        """

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}") as progress_bar:
                for inputs, targets in progress_bar:
                    inputs, targets = inputs.unsqueeze(0).to(self.device), targets.unsqueeze(0).to(self.device)

                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(targets[:,:-1],inputs)
                    # Compute and backpropagate loss
                    outputs = outputs.squeeze(0)
                    targets = targets.squeeze(0)[1:]
                    loss = self.criterion(outputs, targets)
                    total_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()
                    progress_bar.set_postfix({"loss": loss.item()})
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Average loss: {avg_loss:.4f}")
            if (epoch % 10 == 0):
                torch.save(self.model.state_dict(), f"../models/{self.name}_{epoch + 1}.pth")
            # Validation
            self.validate(val_loader)

    def validate(self, val_loader):
        """
        Validates the model on the given dataset.

        Args:
            val_loader: DataLoader for the validation data.
        """
        self.model.eval()
        val_loss = 0
        with tqdm(val_loader, desc=f"Validating") as progress_bar:
            with torch.no_grad():
                for inputs, targets in progress_bar:
                    inputs, targets = inputs.unsqueeze(0).to(self.device), targets.unsqueeze(0).to(self.device)          
                    outputs = self.model(targets[:,:-1],inputs)
                    # Compute and backpropagate loss
                    outputs = outputs.squeeze(0)
                    targets = targets.squeeze(0)[1:]
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()


        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")



    def generate_summary(self, tokenizer, inputs, max_length=1024, n_gram_blocking=3): 
        """
        Generate a summary for a given input text using the model.
        Args:
            tokenizer: The tokenizer for the model.
            inputs: Encoded input tensor.
            max_length: The maximum length of the generated summary.
            n_gram_blocking: The size of n-grams to block from repetition.
        """
        self.model.eval()

        bos_token_id = tokenizer.bos_token_id  # [START] token id
        eos_token_id = tokenizer.eos_token_id  # [END] token id

        with torch.no_grad():
            decoder_input_ids = torch.tensor([bos_token_id], dtype=torch.long).unsqueeze(0).to(self.device)  # (1, n)
            inputs = inputs.unsqueeze(0).to(self.device)
            generated_ngrams = set()  # To store generated n-grams

            for i in range(max_length):       
                outputs = self.model(decoder_input_ids, inputs)
                outputs = outputs.squeeze(0)  # (n_steps, vocab_size)

                # Last hidden states
                logits = outputs[-1, :]

                # Predict probability
                probs = torch.nn.functional.softmax(logits, dim=-1) # [vocab_size]

                # Apply n-gram blocking
                if decoder_input_ids.size(1) >= n_gram_blocking - 1:
                    # Extract the most recent (n-1)-gram
                    recent_ngram = tuple(decoder_input_ids[0, -(n_gram_blocking - 1):].tolist())
                    # Add matching next-token possibilities to the block list
                    for token_id in range(probs.size(0)):
                        candidate_ngram = recent_ngram + (token_id,)
                        if candidate_ngram in generated_ngrams:
                            probs[token_id] = 0  # Set probability to zero for blocked tokens

                # Normalize probabilities after n-gram blocking

                probs = probs / probs.sum()


                # Sample the next token
                next_token_id = torch.argmax(probs, dim=-1).unsqueeze(0).unsqueeze(0)  # Greedy decoding

                # If the generated token is the [END] token, stop the process
                if next_token_id.item() == eos_token_id:
                    break

                # Update decoder input and n-grams
                decoder_input_ids = torch.cat((decoder_input_ids, next_token_id), dim=1)
                if decoder_input_ids.size(1) >= n_gram_blocking:
                    # Add the new n-gram to the set
                    new_ngram = tuple(decoder_input_ids[0, -n_gram_blocking:].tolist())
                    generated_ngrams.add(new_ngram)

            summary_text = tokenizer.decode(decoder_input_ids[0].tolist(), skip_special_tokens=True)
        
            return summary_text
        


