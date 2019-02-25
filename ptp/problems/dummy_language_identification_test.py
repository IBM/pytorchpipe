
import logging

import torch.optim as optim
from torch.utils.data import DataLoader

from ptp.utils.param_interface import ParamInterface
from ptp.utils.problem_manager import ProblemManager
from ptp.utils.pipeline_manager import PipelineManager



if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""
    # Set logging.
    logging.basicConfig(level=logging.INFO)

    # "Simulate" configuration.
    params = ParamInterface()
    params.add_config_params({
        'training': {
            'problem': {
                'type': 'DummyLanguageIdentification',
                'priority': 1,
                'data_folder': '~/data/language_identification/dummy',
                'use_train_data': True,
                'keymappings' : {'inputs': 'sentences', 'targets': 'languages'},
                'batch_size': 2
            }
        },
        'pipeline': {
            #'skip': 'sentence_tokenizer',
            # Sentences encoding.
            'sentence_tokenizer': {
                'type': 'SentenceTokenizer',
                'priority': 2,
                'keymappings' : {'inputs': 'sentences', 'outputs': 'tokenized_sentences'}
            },
            'sentence_encoder': {
                'type': 'SentenceEncoder',
                'priority': 3,
                'data_folder': '~/data/language_identification/dummy',
                'source_files': 'x_training.txt,x_test.txt',
                'encodings_file': 'word_encodings.csv',
                'keymappings' : {
                    'inputs': 'tokenized_sentences',
                    'outputs': 'encoded_sentences'
                    }
            },
            'bow_encoder': {
                'type': 'BOWEncoder',
                'priority': 4,
                'keymappings' : {
                    'inputs': 'encoded_sentences',
                    'outputs': 'bow_sencentes',
                    'input_size': 'sentence_token_size', # Set by sentence_encoder.
                    }
            },
            # Targets encoding.
            'label_encoder': {
                'type': 'LabelEncoder',
                'priority': 5,
                'data_folder': '~/data/language_identification/dummy',
                'source_files': 'y_training.txt,y_test.txt',
                'encodings_file': 'language_name_encodings.csv',
                'keymappings' : {
                    'inputs': 'languages',
                    'outputs': 'encoded_languages'
                    }
            },
            # Model
            'model': {
                'type': 'SoftmaxClassifier',
                'priority': 6,
                'keymappings' : {
                    'inputs': 'bow_sencentes',
                    #'predictions': 'encoded_predictions',
                    'input_size': 'sentence_token_size', # Set by sentence_encoder.
                    'prediction_size': 'label_token_size' # Set by target_encoder.
                    }
            },
            # Loss
            'nllloss': {
                'priority': 7,
                'type': 'NLLLoss',
                'keymappings' : {
                    'targets': 'encoded_languages',
                    #'predictions': 'encoded_predictions',
                    'loss': 'loss'
                    }
            },
            # Predictions decoder.
            'prediction_decoder': {
                'priority': 8,
                'type': 'WordDecoder',
                'data_folder': '~/data/language_identification/dummy',
                'encodings_file': 'language_name_encodings.csv',
                'keymappings' : {'inputs': 'predictions', 'outputs': 'predicted_labels'}
            }
        } #: pipeline
        })


    #### Pipeline "configuration" ####
    errors = 0

    # Build problem.
    prob_mgr = ProblemManager("training", params["training"])
    errors += prob_mgr.build("problem")

    # Build pipeline.
    pipe_mgr = PipelineManager(params["pipeline"])
    errors += pipe_mgr.build()

    # Show pipeline.
    summary_str = pipe_mgr.summarize_io_header()
    summary_str += prob_mgr.problem.summarize_io()
    summary_str += pipe_mgr.summarize_io()
    print(summary_str)

    # Handshake definitions.
    defs = prob_mgr.problem.output_data_definitions()
    errors += pipe_mgr.handshake(defs)

    # Check errors.
    if errors > 0:
        exit(1)


    # Get dataloader.
    dataloader = prob_mgr.loader

    optimizer = optim.SGD(pipe_mgr.parameters(), lr=0.1)

    # Usually you want to pass over the training data several times.
    # 100 is much bigger than on a real data set, but real datasets have more than
    # two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
    for epoch in range(100):
        for i, batch in enumerate(dataloader):
            # Step 1. Remember that PyTorch accumulates gradients.
            # We need to clear them out before each instance
            pipe_mgr.zero_grad()

            # Process batch.
            pipe_mgr.forward(batch)

            print("sequences: {} \t\t targets: {}  ->  predictions: {}".format(batch["sentences"], batch["languages"], batch["predicted_labels"]))

            # Step 4. Compute the loss, gradients, and update the parameters by
            pipe_mgr.backward(batch)
            optimizer.step()

    # Print last batch.
    # print(batch)