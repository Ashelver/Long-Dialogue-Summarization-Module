# Long-Dialogue-Summarization-Module
A model designed for long dialogue summarization

## Requirements:
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install json
pip install transformers -U
pip install accelerate -U
pip install sentencepiece
pip install scikit-learn
```


## Referenced paper: 

    "Yilun Hua, Zhaoyuan Deng, and Kathleen McKeown. 2023. Improving Long Dialogue Summarization with Semantic Graph Representation. In Findings of the Association for Computational Linguistics: ACL 2023, pages 13851–13883, Toronto, Canada. Association for Computational Linguistics."


## Dataset Download (AMI and ICSI corpora transcripts):

    The paper doesn't provide dataset link, so I used the following dataset download link: https://github.com/guokan-shang/ami-and-icsi-corpora?tab=readme-ov-file
    
    Unzip the downloaded two compressed packages to the 'raw_data' directory




## Code Execution Workflow:

1. **Download and Prepare the Dataset**:
   - Download the necessary datasets (`ami-corpus` and `icsi-corpus`).
   - Unzip the datasets and place them in the `raw_data` directory:
     - `raw_data/ami-corpus`
     - `raw_data/icsi-corpus`

2. **Run the Training Scripts**:
   - Navigate to the `src` directory (this is important!).
   - Run the following scripts:
     - `train_ami.py`
     - `train_icsi.py`
   - After running the scripts, the following will be generated:
     - Model files: 
       - `ami_141.pth`
       - `icsi_141.pth`
     - Tokenizer files for both `ami` and `icsi`.
     - Processed data files in `./processed_data`:
       - `ami.json`
       - `icsi.json`

3. **Generate Output Data**:
   - In the `./results` directory, you will find:
     - `ami.json` containing the generated summaries and target summaries for AMI dataset.
     - `icsi.json` containing the generated summaries and target summaries for ICSI dataset.

4. **Run ROUGE Evaluation**:
   - After the model training and summary generation, run the `rouge_evaluate.py` script to evaluate the ROUGE scores for the generated summaries.
   - The ROUGE scores for both the `ami` and `icsi` datasets will be printed in the terminal.





