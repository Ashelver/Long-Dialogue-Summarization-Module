import torch
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from data_preprocessing import DialogueProcessor
from tokens_extracting import DialogueTokensProcessor
from GRUTransformer import GRUConvTransformer



if __name__ == "__main__":
    
    directory = '../raw_data/ami-corpus/'
    processor = DialogueProcessor(directory)
    # Process and save the results to a specified JSON file
    output_file = '../processed_data/ami.json'
    processor.save_to_json(output_file)

    print(f"Processing results saved to {output_file}")

    file_path = '../processed_data/ami.json'
    processor = DialogueTokensProcessor(name='ami')
    vocab_size, embeddings, tgts_ids = processor.process_file(file_path)

    src_train, src_valid, target_train, target_valid = train_test_split(
    embeddings, tgts_ids, test_size=0.25, random_state=42  # train:valid = 3:1
    )

    train_data = list(zip(src_train, target_train))
    valid_data = list(zip(src_valid, target_valid))
    all_data = list(zip(embeddings, tgts_ids))

    model = GRUConvTransformer(
        name='ami',
        input_dim=768, 
        gru_hidden_dim=1024,
        transformer_d_model=1024, 
        num_heads=8,
        num_decoder_layers=12, 
        learning_rate=1e-5, 
        vocab_size=vocab_size
    )

    model.train(train_data, valid_data, epochs=150)

    tokenizer = BertTokenizer.from_pretrained('../models/ami_tokenizer')

    outputs = []
    with tqdm(valid_data, desc="Generating summaries:") as progress_bar:
        for input, tgt in progress_bar:
            output_target = {}
            output_target["output"] = model.generate_summary(tokenizer=tokenizer, inputs=input)
            output_target["target"] = processor.decode(tgt)
            outputs.append(output_target)

    with open("../results/ami.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)
