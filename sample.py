import json
import os
import random
import tensorflow as tf
import keras
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras.utils import to_categorical
import joblib
import config


class VideoDescriptionTrain(object):
    def __init__(self, config):
        self.train_path = config.train_path
        self.test_path = config.test_path
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.lr = config.learning_rate
        self.epochs = config.epochs
        self.latent_dim = config.latent_dim
        self.validation_split = config.validation_split
        self.num_encoder_tokens = config.num_encoder_tokens
        self.num_decoder_tokens = config.num_decoder_tokens
        self.time_steps_encoder = config.time_steps_encoder
        self.time_steps_decoder = None
        self.x_data = {}
        self.tokenizer = None
        self.encoder_model = None
        self.decoder_model = None
        self.inf_encoder_model = None
        self.inf_decoder_model = None
        self.save_model_path = config.save_model_path

    def preprocessing(self):
        """Preprocessing the data"""
        TRAIN_LABEL_PATH = os.path.join(self.train_path, 'training_label.json')
        with open(TRAIN_LABEL_PATH) as data_file:
            y_data = json.load(data_file)
        
        train_list = []
        vocab_list = []
        for y in y_data:
            for caption in y['caption']:
                caption = "<bos> " + caption + " <eos>"
                if len(caption.split()) <= 10 and len(caption.split()) >= 6:
                    train_list.append([caption, y['id']])

        random.shuffle(train_list)
        split_idx = int(len(train_list) * self.validation_split)
        training_list = train_list[split_idx:]
        validation_list = train_list[:split_idx]
        
        vocab_list = [train[0] for train in training_list]
        self.tokenizer = Tokenizer(num_words=self.num_decoder_tokens)
        self.tokenizer.fit_on_texts(vocab_list)

        TRAIN_FEATURE_DIR = "D:/ml proj/Video-Captioning-main/data/training_data/feat"
        for filename in os.listdir(TRAIN_FEATURE_DIR):
            print(filename)
            f = np.load(os.path.join(TRAIN_FEATURE_DIR, filename), allow_pickle=True)
            # Pad or truncate the features to match time_steps_encoder
            if f.shape[0] < self.time_steps_encoder:
                pad_width = ((0, self.time_steps_encoder - f.shape[0]), (0, 0))
                f = np.pad(f, pad_width, mode='constant')
            else:
                f = f[:self.time_steps_encoder]
            self.x_data[filename[:-4]] = f

        return training_list, validation_list

    def prepare_dataset(self, data_list):
        """Prepare dataset with correct shapes"""
        encoder_inputs = []
        decoder_inputs = []
        decoder_targets = []
        
        # Process all sequences first
        video_sequences = [cap[0] for cap in data_list]
        train_sequences = self.tokenizer.texts_to_sequences(video_sequences)
        train_sequences = pad_sequences(train_sequences, padding='post', 
                                     truncating='post', maxlen=self.time_steps_encoder)
       # print(data_list)
        # Prepare data in batches
        for idx, cap in enumerate(data_list):
            video_features = self.x_data[cap[1]]
            encoder_inputs.append(video_features)
            #print(encoder_inputs)
            # Convert sequence to one-hot encoding
            sequence = train_sequences[idx]
            y = to_categorical(sequence, self.num_decoder_tokens)
            
            # Prepare decoder inputs (shifted left by one position)
            decoder_input = np.zeros((self.time_steps_encoder, self.num_decoder_tokens))
            decoder_input[1:] = y[:-1]
            decoder_inputs.append(decoder_input)
            
            # Prepare decoder targets (shifted right by one position)
            decoder_target = np.zeros((self.time_steps_encoder, self.num_decoder_tokens))
            decoder_target[:-1] = y[1:]
            decoder_targets.append(decoder_target)

        # Convert lists to numpy arrays with correct shapes
        encoder_inputs = np.array(encoder_inputs)
        decoder_inputs = np.array(decoder_inputs)
        decoder_targets = np.array(decoder_targets)
        # print("decoder_inputs:",decoder_inputs)
        # print("encoder_inputs:",encoder_inputs)
        return (encoder_inputs, decoder_inputs), decoder_targets

    def train_model(self):
        """Train the encoder-decoder sequence model"""
        # Create the model with correct input shapes
        encoder_inputs = Input(shape=(self.time_steps_encoder, self.num_encoder_tokens))
        encoder = LSTM(self.latent_dim, return_state=True, return_sequences=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(self.time_steps_encoder, self.num_decoder_tokens))
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, state1_h, state1_c = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_states= [state1_h,state1_c]

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Prepare data
        training_list, validation_list = self.preprocessing()
        train_data, train_targets = self.prepare_dataset(training_list)
        val_data, val_targets = self.prepare_dataset(validation_list)
        print("train_targets:",train_data)
        # Print shapes for verification
        # print("Encoder input shape:", train_data[0].shape)
        # print("Decoder input shape:", train_data[1].shape)
        # print("Decoder target shape:", train_targets.shape)

        # Compile model
        opt = keras.optimizers.Adam(learning_rate=0.0003)
        model.compile(
            metrics=['accuracy'],
            optimizer=opt,
            loss='categorical_crossentropy'
        )

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=4,
            verbose=1,
            mode='min'
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=5,
            verbose=0,
            mode="auto"
        )

        # Train the model
        # model.fit(
        #     train_data,
        #     train_targets,
        #     validation_data=(val_data, val_targets),
        #     epochs=self.epochs,
        #     batch_size=self.batch_size,
        #     callbacks=[reduce_lr, early_stopping]
        # )

        # Save the models
        # if not os.path.exists(self.save_model_path):
        #     os.makedirs(self.save_model_path)

        # Create and save inference models
        # self.encoder_model = Model(encoder_inputs, encoder_states)
        # #self.decoder_model = Model(decoder_inputs, decoder_states)

        # decoder_state_input_h = Input(shape=(self.latent_dim,))
        # decoder_state_input_c = Input(shape=(self.latent_dim,))
        # decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        # decoder_outputs, state_h, state_c = decoder_lstm(
        #     decoder_inputs, initial_state=decoder_states_inputs)
        # decoder_states = [state_h, state_c]
        # decoder_outputs = decoder_dense(decoder_outputs)
        
        # self.decoder_model = Model(
        #     [decoder_inputs] + decoder_states_inputs,
        #     [decoder_outputs] + decoder_states
        # )

        # # Save models and tokenizer
        # self.encoder_model.save(os.path.join(self.save_model_path, 'encoder_model.h5'))
        # self.decoder_model.save_weights(os.path.join(self.save_model_path, 'decoder_model.weights.h5'))
        # self.decoder_model.save(os.path.join(self.save_model_path, 'decoder_model.keras'))
        
        # tokenizer_path = os.path.join(self.save_model_path, f'tokenizer{self.num_decoder_tokens}')
        # with open(tokenizer_path, 'wb') as file:
        #     joblib.dump(self.tokenizer, file)


if __name__ == "__main__":
    video_to_text = VideoDescriptionTrain(config)
    video_to_text.train_model()
    #video_to_text.preprocessing()