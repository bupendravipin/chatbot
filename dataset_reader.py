'''
dataset reader class
'''
import os,math
import numpy as np
from vocabulary import Vocabulary

class Datasetreader:
    def __init__(self,dataset_dir):
        self.dataset_dir=dataset_dir
        
    def get_movie_dialog_conversations(self,dataset_dir):
        
        """Get dialog lines and conversations.

        Args:
            dataset_dir: path of dataset_old directory
        """
        movie_lines_filepath = os.path.join(dataset_dir, "movie_lines.txt")
        movie_conversations_filepath = os.path.join(dataset_dir, "movie_conversations.txt")
        
        # Importing the dataset_old
        with open(movie_lines_filepath, encoding="utf-8", errors="ignore") as file:
            lines = file.read().split("\n")
        
        with open(movie_conversations_filepath, encoding="utf-8", errors="ignore") as file:
            conversations = file.read().split("\n")
        
        # Creating a dictionary that maps each line and its id
        id2line = {}
        for line in lines:
            _line = line.split(" +++$+++ ")
            if len(_line) == 5:
                id2line[_line[0]] = _line[4]
        
        # Creating a list of all of the conversations
        conversations_ids = []
        for conversation in conversations[:-1]:
            _conversation = conversation.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
            conv_ids = _conversation.split(",")
            conversations_ids.append(conv_ids)
        
        return id2line, conversations_ids
        
    def read_dataset(self,dataset_dir,model_dir,training_hparams, share_vocab):
        if share_vocab and training_hparams.input_vocab_threshold != training_hparams.output_vocab_threshold:
            raise ValueError("Cannot share vocabulary when the input and output vocab thresholds are different.")
        id2line, conversations_ids = self.get_movie_dialog_conversations(dataset_dir)
        #Clean dialog lines
        for line_id in id2line:
            id2line[line_id] = Vocabulary.clean_text(id2line[line_id], training_hparams.max_question_answer_words)
                
        # Getting separately the questions and the answers
        questions_for_count = []
        questions = []
        answers = []
        for conversation in conversations_ids[:training_hparams.max_conversations]:
            for i in range(len(conversation) - 1):
                conv_up_to_question = ''
                for j in range(max(0, i - training_hparams.conv_history_length), i):
                    conv_up_to_question += id2line[conversation[j]] + " {0} ".format(Vocabulary.EOS)
                question = id2line[conversation[i]]
                question_with_history = conv_up_to_question + question
                answer = id2line[conversation[i+1]]
                if training_hparams.min_question_words <= len(question_with_history.split()):
                    questions.append(conv_up_to_question + question)
                    questions_for_count.append(question)
                    answers.append(answer)

        # Create the vocabulary object & add the question & answer words
        if share_vocab:
            questions_and_answers = []
            for i in range(len(questions_for_count)):
                question = questions_for_count[i]
                answer = answers[i]
                if i == 0 or question != answers[i - 1]:
                    questions_and_answers.append(question)
                questions_and_answers.append(answer)
            questions_and_answers=questions_and_answers[:10]
            input_vocabulary = self.create_and_save_vocab(questions_and_answers, training_hparams.input_vocab_threshold, model_dir, Vocabulary.SHARED_VOCAB_FILENAME)
            output_vocabulary = input_vocabulary
        else:
            #different vocabularies for input and output
            input_vocabulary = self.create_and_save_vocab(questions_for_count, training_hparams.input_vocab_threshold, model_dir, Vocabulary.INPUT_VOCAB_FILENAME)
            output_vocabulary = self.create_and_save_vocab(answers, training_hparams.output_vocab_threshold, model_dir, Vocabulary.OUTPUT_VOCAB_FILENAME)
        
        # Adding the End Of String tokens to the end of every answer
        for i in range(len(answers)):
            answers[i] += " {0}".format(Vocabulary.EOS)
        

        return questions, answers, input_vocabulary, output_vocabulary
    
    def create_and_save_vocab(self, word_sequences, vocab_threshold, model_dir, vocab_filename):
        """Create a Vocabulary instance from a list of word sequences, and save it to disk.

        Args:
            word_sequences: List of word sequences (sentence(s)) to use as basis for the vocabulary.

            vocab_threshold: Minimum number of times any word must appear within word_sequences 
                in order to be included in the vocabulary.
        """
        vocabulary = Vocabulary()
        for i in range(len(word_sequences)):
            word_seq = word_sequences[i]
            vocabulary.add_words(word_seq.split())
        vocabulary.compile(vocab_threshold)

        vocab_filepath = os.path.join(model_dir, vocab_filename)
        vocabulary.save(vocab_filepath)
        return vocabulary
    
    def train_val_split(self,questions,answers,input_vocab,output_vocab,test_percent):
        """Splits the dataset_old into training and validation sets.
        
        Args:
            test_percent: the percentage of the dataset_old to use as validation data.
        """
        questions = questions[:]
        answers = answers[:]
        num_validation_samples = int(len(questions) * (test_percent / 100))
        num_training_samples = len(questions) - num_validation_samples
        training_questions = []
        training_answers = []
        validation_questions = []
        validation_answers = []
        
        for _ in range(num_training_samples):
            training_questions.append(questions.pop(0))
            training_answers.append(answers.pop(0))
            
        for _ in range(num_validation_samples):
            validation_questions.append(questions.pop(0))
            validation_answers.append(answers.pop(0))
            
        training_dataset_questions_into_int, training_dataset_answers_into_int = self.validate_questions_answers(training_questions, training_answers, input_vocab, output_vocab)
        validation_dataset_questions_into_int,validation_dataset_answers_into_int = self.validate_questions_answers(validation_questions, validation_answers, input_vocab, output_vocab)
        
        return training_dataset_questions_into_int, training_dataset_answers_into_int,validation_dataset_questions_into_int,validation_dataset_answers_into_int
    
    def validate_questions_answers(self,questions,answers,input_vocabulary,output_vocabulary):
        if len(questions) != len(answers):
            raise RuntimeError("questions and answers lists must be the same length")
        #If the questions and answers are already integer encoded, accept them as is.
        #Otherwise use the Vocabulary instances to encode the question and answer sequences.
        if len(questions) > 0 and isinstance(questions[0], str):
            questions_into_int = [input_vocabulary.words2ints(q) for q in questions]
            answers_into_int = [output_vocabulary.words2ints(a) for a in answers]
        else:
            questions_into_int = questions
            answers_into_int = answers
        return questions_into_int, answers_into_int
    
    def sort(self,questions_into_int, answers_into_int):
        """Sorts the dataset_old by the lengths of the questions. This can speed up training by reducing the
        amount of padding the input sequences need.
        """
        questions_into_int, answers_into_int = zip(*sorted(zip(questions_into_int, answers_into_int), 
                                                                         key = lambda qa_pair: len(qa_pair[0])))
        return questions_into_int, answers_into_int
    
    def batches(self,questions_into_int,answers_into_int,input_vocabulary,output_vocabulary,batch_size):
        
        for batch_index in range(0, math.ceil(len(questions_into_int) / batch_size)):
                start_index = batch_index * batch_size
                questions_in_batch = questions_into_int[start_index : start_index + batch_size]
                answers_in_batch = answers_into_int[start_index : start_index + batch_size]
                
                seqlen_questions_in_batch = np.array([len(q) for q in questions_in_batch])
                seqlen_answers_in_batch = np.array([len(a) for a in answers_in_batch])
                
                padded_questions_in_batch = np.array(self.apply_padding(questions_in_batch, input_vocabulary))
                padded_answers_in_batch = np.array(self.apply_padding(answers_in_batch, output_vocabulary))
                
                yield padded_questions_in_batch, padded_answers_in_batch, seqlen_questions_in_batch, seqlen_answers_in_batch
                
    def apply_padding(self,batch_of_sequences, vocabulary):
        max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
        return [sequence + ([vocabulary.pad_int()] * (max_sequence_length - len(sequence))) for sequence in batch_of_sequences]

        