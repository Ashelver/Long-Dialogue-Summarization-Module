import os
import re
import json

class DialogueProcessor:
    def __init__(self, directory):
        """
        Initialize the class with the directory containing the data
        :param directory: The root directory containing 'abstractive' and 'dialogueActs' subdirectories
        """
        self.directory = directory
        self.src_dir = os.path.join(directory, 'dialogueActs')
        self.tgt_dir = os.path.join(directory, 'abstractive')
        self.max_speech_length = 0
        self.max_speaker_turns = 0
        self.total_speech_count = 0
        self.remove_words = ["Uh-huh","Mm-hmm","Mm ","Hmm ", "Mm - hmm", "Mmm ", "Um "]

    def remove_word_from_sentence(self, sentence):
        """
        Remove the specified word from the sentence. If the sentence becomes empty 
        (after removing spaces and punctuation), return None.

        :param sentence: The input sentence from which the word will be removed.
        :param word_to_remove: The word to be removed from the sentence.
        :return: The modified sentence or None if the sentence becomes empty.
        """
        for word_to_remove in self.remove_words:
            # Remove the word from the sentence, case-insensitive
            sentence = re.sub(r'\b' + re.escape(word_to_remove) + r'\b', '', sentence, flags=re.IGNORECASE)

            # Remove any HTML/XML-like tags (e.g., <vocalsound>)
            sentence = re.sub(r'<.*?>', '', sentence)

            # Remove trailing punctuation like periods
            sentence = sentence.rstrip('.')

            # Remove commas at the beginning of the sentence
            sentence = re.sub(r'^\s*,\s*', '', sentence)
            
            # Remove full stops at the beginning of the sentence
            sentence = re.sub(r'^\s*\.\s*', '', sentence)

            # Remove extra spaces from the sentence
            sentence = re.sub(r'\s+', ' ', sentence).strip()

            # If sentence is empty after removing the word and cleaning up, return None
            if len(sentence.replace(' ', '').replace('.', '')) == 0:
                return None

        return sentence


    def process_dialogue(self, src_data, tgt_data):
        """
        Process each pair of dialogue and abstractive data to generate the 'src' and 'tgt' format
        :param src_data: Source data containing dialogue information
        :param tgt_data: Target data containing abstract summary information
        :return: A dictionary {'src': src_text, 'tgt': tgt_text}
        """
        # Build src: create a list of sentences, combining consecutive sentences from the same speaker
        src = []
        current_speaker = None
        current_sentence = []

        current_sentence_length = 0  # Track the max length of a speech
        speaker_turns = {}  # Track the number of turns per speaker

        for entry in src_data:
            speaker = entry['speaker']
            text = self.remove_word_from_sentence(entry['text'])
            
            if text is None:
                continue

            if speaker != current_speaker or current_sentence_length > 50:
                if current_sentence:
                    # If there's an ongoing sentence from the previous speaker, add it to the list
                    src.append(' '.join(current_sentence))
                # Start a new sentence for the current speaker
                current_sentence = [f"{speaker}: {text}"]
                current_sentence_length = len(text.split())
                # Update speaker turn count
                speaker_turns[speaker] = speaker_turns.get(speaker, 0) + 1
                self.total_speech_count += 1
            else:
                # Continue the sentence from the same speaker
                current_sentence.append(text)
                current_sentence_length += len(text.split())

            # Track the speech length
            self.max_speech_length = max(self.max_speech_length, current_sentence_length)

            current_speaker = speaker
            

        if current_sentence:
            src.append(' '.join(current_sentence))  # Add the last sentence

        # Build tgt: concatenate all abstract summary texts into a list
        tgt = [entry['text'] for entry in tgt_data]

        # Calculate max number of speeches by any speaker
        max_speaker_turns = max(speaker_turns.values(), default=0)

        # Accumulate statistics from all dialogues
        self.max_speaker_turns = max(self.max_speaker_turns, max_speaker_turns)

        return {"text": src, "tgt": tgt}

    def process_directory(self):
        """
        Traverse the directory, process each pair of json files
        """
        result = []
        
        # Get all the file names (without extensions)
        src_files = os.listdir(self.src_dir)
        
        for src_file in src_files:
            if src_file.endswith(".json"):
                # Find the corresponding tgt file
                tgt_file = src_file
                src_file_path = os.path.join(self.src_dir, src_file)
                tgt_file_path = os.path.join(self.tgt_dir, tgt_file)

                if os.path.exists(tgt_file_path):
                    # Read the src and tgt data
                    with open(src_file_path, 'r') as f:
                        src_data = json.load(f)

                    with open(tgt_file_path, 'r') as f:
                        tgt_data = json.load(f)

                    # Process the data
                    processed_data = self.process_dialogue(src_data, tgt_data)
                    result.append(processed_data)

        return result

    def save_to_json(self, output_file):
        """
        Save the processed data as a JSON file
        :param output_file: Path to the output file
        """
        processed_data = self.process_directory()
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=4)

        # Print statistics after processing
        self.print_statistics()

    def print_statistics(self):
        """
        Print statistics about the dialogue
        :param processed_data: The processed data from the dialogues
        """
        print(f"Max speech length: {self.max_speech_length} words")
        print(f"Max number of speeches by any speaker: {self.max_speaker_turns} turns")
        print(f"Total number of speeches across all dialogues: {self.total_speech_count}")


