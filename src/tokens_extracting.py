import torch
from transformers import BertTokenizer, BertModel
import json
from tqdm import tqdm

class DialogueTokensProcessor:
    def __init__(self, name='unknown', tokenizer='bert-base-uncased', max_len=256, overlap=128):
        """
        Initialize the DialogueProcessor with BERT tokenizer and model.
        :param tokenizer: Name of the pre-trained BERT model
        :param max_len: Maximum length of each window for sliding window tokenization
        :param overlap: Overlap between consecutive windows in tokenization
        """
        # Load BERT tokenizer and model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print("Using cuda to get the tokens.")
        else:
            print("Using cpu to get the tokens.")

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        if tokenizer == 'bert-base-uncased':
            special_tokens_dict = {
                "bos_token": "[BOS]",
                "eos_token": "[EOS]"
            }
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.tokenizer.save_pretrained(f'../models/{name}_tokenizer')

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.resize_token_embeddings(len(self.tokenizer))
        self.bert_model.eval().to(self.device)
        self.max_len = max_len
        self.overlap = overlap



    def sliding_window_tokenize(self, text):
        """
        Tokenizes long text into multiple windows using sliding window technique.
        :param text: The input text to tokenize
        :return: A list of token ID windows
        """
        tokens = self.tokenizer.tokenize(text)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        windows = []
        start = 0

        while start < len(input_ids):
            end = min(start + self.max_len, len(input_ids))
            window = [self.tokenizer.cls_token_id] + input_ids[start:end] + [self.tokenizer.sep_token_id]
            windows.append(window)
            # Update start, ensuring it progresses without going backward
            start += self.overlap
            if start >= len(input_ids):  # Stop if start exceeds the input length
                break

        return windows

    def process_dialogue(self, dialogue):
        """
        Process the dialogue data (text and tgt) and apply sliding window tokenization.
        :param dialogue: A dictionary containing 'text' (list of strings) and 'tgt' (string)
        :return: A list of BERT embeddings for each window of the text and token ids of both text and tgt
        """
        # Concatenate all text parts into one string
        text = " ".join(dialogue["text"])  
        tgt = " ".join(dialogue["tgt"] )

        # Tokenize text and target
        tgt_tokens = self.tokenizer.tokenize(tgt)    # Tokenize the target

        # Convert tokens to IDs
        tgt_input_ids = self.tokenizer.convert_tokens_to_ids(tgt_tokens)
        sos_token_id = self.tokenizer.bos_token_id  # Add start symbol
        eos_token_id = self.tokenizer.eos_token_id  # Add start symbol
        tgt_input_ids = [sos_token_id] + tgt_input_ids + [eos_token_id]


        windows = self.sliding_window_tokenize(text)
        
        bert_embeddings = []
        # Process each window using BERT
        for _, window in enumerate(windows):  # For each window in this dialogue
            input_ids = torch.tensor([window]).to(self.device)  # Move input IDs to GPU
            with torch.no_grad():
                outputs = self.bert_model(input_ids)
            bert_embeddings.append(outputs.last_hidden_state[:, 0, :])  # Get [CLS] token embedding

        # Stack all embeddings and return
        return torch.stack(bert_embeddings).squeeze(1), torch.tensor(tgt_input_ids, dtype=torch.long)

    def process_file(self, file_path):
        """
        Process the JSON file with dialogues and return the BERT embeddings for each dialogue.
        :param file_path: Path to the JSON file containing the dialogues
        :return: A list of BERT embeddings for all dialogues
        """
        with open(file_path, 'r') as f:
            dialogues = json.load(f)

        max_lenth_tgt = 0
        embeddings = []
        tgts_ids = []
        # Print the progress
        for dialogue in tqdm(dialogues, desc="Processing dialogues", unit="dialogue"):
            bert_embeddings, tgt_ids = self.process_dialogue(dialogue)
            if max_lenth_tgt < tgt_ids.shape[0]:
                max_lenth_tgt = tgt_ids.shape[0]

            # Append the results
            embeddings.append(bert_embeddings)
            tgts_ids.append(tgt_ids)

        print("Max target length:",max_lenth_tgt)

        return len(self.tokenizer), embeddings, tgts_ids


    def decode(self, id_tensor):
        """
        Decode a list of token IDs into text and save to a file.
        :param ids_tensor: Tensors of token IDs to decode
        :param output_file: File path to save the decoded text
        """
        decoded_text = self.tokenizer.decode(id_tensor.tolist(), skip_special_tokens=True)
        return decoded_text



