# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 07:13:17 2020

@author: ljh
"""
from abc import ABC
import json
import logging
import os
import traceback

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer,AutoConfig

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TransformersClassifierHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        
        print("initialize________:",self.manifest)

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
  
        # Read model serialize/pt file

        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        logger.debug('Transformer initialize tokenizer： {0}'.format(model_dir))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)


        # self.model.to(self.device)
        self.model.eval()


        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')

        self.initialized = True
        print("initialize_____initialized___OK")

    def preprocess(self, data):
        """ Very basic preprocessing code - only tokenizes. 
            Extend with your own preprocessing steps as needed.
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentences = text.decode('utf-8')
        logger.info("Received text: '%s'", sentences)
        print(sentences)

        inputs = self.tokenizer.encode_plus(
            sentences,
            add_special_tokens=True,
            return_tensors="pt"
        )
        return inputs

    def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """
        # NOTE: This makes the assumption that your model expects text to be tokenized  
        # with "input_ids" and "token_type_ids" - which is true for some popular transformer models, e.g. bert.
        # If your transformer model expects different tokenization, adapt this code to suit 
        # its expected input format.

        
        prediction = self.model( inputs['input_ids'], 
                                attention_mask = inputs['attention_mask']
        )[0].argmax().item()
        
        logger.info("Model predicted: '%s'", prediction)

        if self.mapping:
            prediction = self.mapping[str(prediction)]

        return [prediction]

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output


_service = TransformersClassifierHandler()


def handle(data, context):
 
    try:

        if not _service.initialized:

            _service.initialize(context)


        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        traceback.print_exc()

        raise e
        
if __name__== "__main__":
    from transformers import AutoConfig,AutoModelForSequenceClassification,AutoTokenizer
    
    config = AutoConfig.from_pretrained(r'./distilbert-base-uncased/')
    modelbert = AutoModelForSequenceClassification.from_pretrained(r'./distilbert-base-uncased/', config=config)
    tokenizer = AutoTokenizer.from_pretrained(r'./distilbert-base-uncased/')
    
#     sentences = 'you are so bad'
#     inputs = tokenizer.encode_plus(
#             sentences,
#             add_special_tokens=True,
#             # return_token_type_ids = True,
#             return_tensors="pt"
#         )
#     modelbert( inputs['input_ids'],  attention_mask = inputs['attention_mask'])

    
    NEW_DIR = 'model_store'
    modelbert.save_pretrained(NEW_DIR)
    tokenizer.save_pretrained(NEW_DIR)
    
    
        
        