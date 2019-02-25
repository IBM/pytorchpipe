
import logging

import torch.optim as optim
from torch.utils.data import DataLoader

from ptp.utils.param_interface import ParamInterface
from ptp.utils.pipeline import Pipeline
from ptp.utils.problem_factory import ProblemFactory



if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""
    # Set logging.
    logging.basicConfig(level=logging.INFO)

    # "Simulate" configuration.
    params = ParamInterface()
    params.add_config_params({
        'problem': {
            'type': 'DummyLanguageIdentification',
            'priority': 1,
            'data_folder': '~/data/language_identification/dummy',
            'use_train_data': True,
            'keymappings' : {'inputs': 'sentences', 'targets': 'languages'}
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

    batch_size = 2


    #### Pipeline "configuration" ####
    errors = 0

    # Build problem.
    problem = ProblemFactory.build("problem", params["problem"])
    if (problem == None):
        errors += 1

    # Build pipeline.
    pipeline = Pipeline()
    errors += pipeline.build_pipeline(params["pipeline"])

    # Show pipeline.
    summary_str = pipeline.summarize_io_header()
    summary_str += problem.summarize_io()
    summary_str += pipeline.summarize_io()
    print(summary_str)

    # Handshake definitions.
    defs = problem.output_data_definitions()
    errors += pipeline.handshake(defs)

    # Check errors.
    if errors > 0:
        exit(1)

    # Get problem, model and loss.
    model = pipeline.models[0]
    loss = pipeline.losses[0]
    

    # Construct dataloader.
    dataloader = DataLoader(dataset=problem, collate_fn=problem.collate_fn,
                            batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Usually you want to pass over the training data several times.
    # 100 is much bigger than on a real data set, but real datasets have more than
    # two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
    for epoch in range(100):
        for i, batch in enumerate(dataloader):
            # Step 1. Remember that PyTorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Process batch.
            pipeline(batch)

            print("sequences: {} \t\t targets: {}  ->  predictions: {}".format(batch["sentences"], batch["languages"], batch["predicted_labels"]))

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss_value = batch["loss"]
            #print("Loss = ", loss)

            loss_value.backward()
            optimizer.step()

    # Print last batch.
    # print(batch)