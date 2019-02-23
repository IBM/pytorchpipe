
import torch.optim as optim
from torch.utils.data import DataLoader


from ptp.utils.pipeline import Pipeline

# Problems.
from ptp.problems.dummy_language_identification import DummyLanguageIdentification

# Sentence encoders.
from ptp.text.sentence_tokenizer import SentenceTokenizer
from ptp.text.sentence_encoder import SentenceEncoder
from ptp.text.bow_encoder import BOWEncoder

# Target encoder.
from ptp.text.label_encoder import LabelEncoder

# Model.
from ptp.models.softmax_classifier import SoftmaxClassifier

# Loss.
from ptp.loss.nll_loss import NLLLoss

# Decoder.
from ptp.text.word_decoder import WordDecoder



if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""
    # Set logging.
    import logging
    logging.basicConfig(level=logging.INFO)

    from ptp.utils.param_interface import ParamInterface
    # "Simulate" configuration.
    params = ParamInterface()
    params.add_config_params({
        'problem': {
            'type': 'LanguageIdentification',
            'priority': 1,
            'data_folder': '~/data/language_identification/dummy',
            'use_train_data': True,
            'keymappings' : {'inputs': 'sentences', 'targets': 'languages'}
        },
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

        })

    batch_size = 2

    pipeline = Pipeline(params)
    pipeline.build()
    print(pipeline.summarize())

    exit(1)
    #### "Configuration" ####
    # Create problem.
    problem  = DummyLanguageIdentification("problem", params["problem"])

    # Input (sentence) encoder.
    sentence_tokenizer = SentenceTokenizer("sentence_tokenizer", params["sentence_tokenizer"])
    sentence_encoder = SentenceEncoder("sentence_encoder", params["sentence_encoder"])
    bow_encoder = BOWEncoder("bow_encoder", params["bow_encoder"])

    # Target encoder.
    target_encoder = LabelEncoder("label_encoder", params["label_encoder"])

    # Model.
    model = SoftmaxClassifier("model", params["model"])

    # Loss.
    loss = NLLLoss("nllloss", params["nllloss"])

    # Decoder.
    prediction_decoder  = WordDecoder("prediction_decoder", params["prediction_decoder"])

    #### "Handshaking" ####
    all_definitions = problem.output_data_definitions()
    #print(all_definitions)
    errors = 0

    errors += sentence_tokenizer.handshake_input_definitions(all_definitions)
    errors += sentence_tokenizer.export_output_definitions(all_definitions)

    errors += sentence_encoder.handshake_input_definitions(all_definitions)
    errors += sentence_encoder.export_output_definitions(all_definitions)
    
    errors += bow_encoder.handshake_input_definitions(all_definitions)
    errors += bow_encoder.export_output_definitions(all_definitions)

    errors += target_encoder.handshake_input_definitions(all_definitions)
    errors += target_encoder.export_output_definitions(all_definitions)

    errors += model.handshake_input_definitions(all_definitions)
    errors += model.export_output_definitions(all_definitions)

    errors += loss.handshake_input_definitions(all_definitions)
    errors += loss.export_output_definitions(all_definitions)

    errors += prediction_decoder.handshake_input_definitions(all_definitions)
    errors += prediction_decoder.export_output_definitions(all_definitions)

    if errors > 0:
        exit(1)
    
    # Log final definition.
    print("Final, handskaked definitions of DataDict used in pipeline: \n{}\n".format(all_definitions))

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

            # Input (sentence) encoder.
            sentence_tokenizer(batch)
            sentence_encoder(batch)
            bow_encoder(batch)

            # Target encoder.
            target_encoder(batch)

            # Model.
            model(batch)

            # Loss.
            loss(batch)

            # Decoder.
            prediction_decoder(batch)

            print("sequences: {} \t\t targets: {}  ->  predictions: {}".format(batch["sentences"], batch["languages"], batch["predicted_labels"]))

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss_value = batch["loss"]
            #print("Loss = ", loss)

            loss_value.backward()
            optimizer.step()

    # Print last batch.
    print(batch)