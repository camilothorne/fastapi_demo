import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.tokenization_utils_base import BatchEncoding
from tabulate import tabulate
import os

class ChemClassifier:

    def __init__(self):
        '''
        Constructor
        '''
        # Get path to this file
        self.__root = os.path.dirname(os.path.abspath(__file__))
        # Set up device
        if torch.cuda.is_available():
            self.__device = torch.device('cuda')
        else:
            self.__device = torch.device('cpu')
        # Path to fine-tuned model and tokenizer
        self.__model_checkpoint = self.__root + "/model"
        # Load the BERT model
        self.__model = AutoModelForSequenceClassification.from_pretrained(
            self.__model_checkpoint + '/classif',
            num_labels = 3,
            output_attentions = False,
            output_hidden_states = False,
        )
        # Load the tokenizer
        self.__tokenizer  = AutoTokenizer.from_pretrained(
            self.__model_checkpoint + '/tokenizer', 
            use_fast=True
        )
        # Class labels
        self.__label_dict = {0: 'Biology', 1:'Chemistry', 2: 'Physics'}


    def __preprocessing(self, 
                      input_text:str, 
                      tokenizer:AutoTokenizer) -> BatchEncoding:
        '''
        Returns <class transformers.tokenization_utils_base.BatchEncoding> 
        with the following fields:
        
        - input_ids:      list of token ids,
        - token_type_ids: list of token type ids,
        - attention_mask: list of indices (0,1) 
          specifying which tokens should considered by the model 
          (return_attention_mask = True).
        
        :param input_text: tenxt to tokenize
        :param tokenizer: tokenizer
        :return: BPE token stream 
        '''
        return tokenizer.encode_plus(
                            input_text,
                            add_special_tokens = True,
                            max_length = 32,
                            truncation=True,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt'
                    )


    def __model_inference(self, 
                        in_model:AutoModelForSequenceClassification, 
                        in_tokenizer:AutoTokenizer, label_dict:dict,
                        in_sentence:str) -> dict:
        '''
        Apply a model and tokenizer to an input sentence.
        
        :param in_model: sequence classifier model
        :param in_sentence: 
        :param label_dict: mapping between class IDs and human readable names
        :param in_tokenizer: input tokenizer
        :return: tuple (probability, predicted_class, in_sentence)
        '''
        # We need Token IDs and Attention Mask for inference on the new sentence
        test_ids = []
        test_attention_mask = []
        # Apply the tokenizer
        encoding = self.__preprocessing(in_sentence, in_tokenizer)
        # Extract IDs and Attention Mask
        test_ids.append(encoding['input_ids'])
        test_attention_mask.append(encoding['attention_mask'])
        test_ids = torch.cat(test_ids, dim = 0)
        test_attention_mask = torch.cat(test_attention_mask, dim = 0)
        # Forward pass, calculate logit predictions
        with torch.no_grad():
            output = in_model(test_ids.to(self.__device), 
                              attention_mask = test_attention_mask.to(self.__device))
        # Predictions and probabilities
        prob = torch.softmax(output.logits.cpu()[0],-1).numpy()[1]
        label = label_dict[np.argmax(output.logits.cpu().numpy()).flatten().item()]
        # Return results
        return {'prob': prob, 'label': label, 'input': in_sentence}


    def get_sentence_class(self, 
                           in_sentence:str) -> dict:
        '''
        Return predictions for input sentence.

        :param in_sentence: input sentence
        :return: tuple (probability, predicted_class, in_sentence)
        '''
        return self.__model_inference(self.__model, 
                                    self.__tokenizer, 
                                    self.__label_dict, 
                                    in_sentence)


    def get_sentence_encoding(self, 
                              in_sentence:str) -> str:
        '''
        Returns tokens, token IDs and attention mask of a input text sample
        as a plain-text table

        :param in_sentence: input sentence
        :return: tokens, tokens displayed as table
        '''
        encoding_dict = self.__preprocessing(in_sentence, self.__tokenizer)
        token_ids = encoding_dict['input_ids']
        attention_masks = encoding_dict['attention_mask']
        token_ids = torch.cat([token_ids], dim = 0)
        attention_masks = torch.cat([attention_masks], dim = 0)
        tokens = self.__tokenizer.tokenize(self.__tokenizer.decode(token_ids[0]))
        token_ids = [i.numpy() for i in token_ids[0]]
        attention = [i.numpy() for i in attention_masks[0]]
        table = np.array([tokens, token_ids, attention]).T
        # Return tokens and table
        return tokens, tabulate(table, 
                    headers = ['Tokens', 'Token IDs', 'Attention Mask'],
                    tablefmt = 'fancy_grid')


# if __name__ == '__main__':
#
#     my_classif = ChemClassifier()
#
#     test_sent_1 = '''
#                 I think I have 100 cm of books on the subject. TL;DR: The problem of consciousness
#                 is universally acknowledged as one of the most important in science, tens of 
#                 thousands of scientists have devoted their careers to chipping away at it, 
#                 numerous Nobel laureates have turned from their original fields to tackle it, 
#                 and to date, no one has solved it.
#                 '''
#     print(my_classif.get_sentence_class(test_sent_1))
#     print(my_classif.get_sentence_encoding(test_sent_1))
#
#     test_Sent_2 = "The problem of consciousness is one of the most important in science."
#     print(my_classif.get_sentence_class(test_sent_2))
#     print(my_classif.get_sentence_encoding(test_sent_2))