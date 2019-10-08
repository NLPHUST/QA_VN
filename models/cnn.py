import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *

class CNN_MATCH():
    def __init__(self,config):
        self.config = config
        self.model = self.build()
    
    def build(self):
        seq1 = Input(name='seq1', shape=[self.config['seq1_maxlen']])
        seq2 = Input(name='seq2', shape=[self.config['seq2_maxlen']])
        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], trainable = self.config['embed_trainable'])

        seq1_embed = embedding(seq1)
        seq1_embed = Dropout(rate=self.config['dropout_rate'])(seq1_embed)
        seq2_embed = embedding(seq2)
        seq2_embed = Dropout(rate=self.config['dropout_rate'])(seq2_embed)

        seq1_rep = self.cnn_layer(seq1_embed,sequence_length=self.config['seq1_maxlen'],embedding_dim=self.config['embed_size'])
        seq2_rep = self.cnn_layer(seq2_embed,sequence_length=self.config['seq2_maxlen'],embedding_dim=self.config['embed_size'])

        final_rep = concatenate([seq1_rep,seq2_rep])
        final_rep = BatchNormalization()(final_rep)
        final_rep = Dropout(rate=self.config['dropout_rate'])(final_rep)

        if self.config['target_mode'] == 'classification':
            out = Dense(2, activation='softmax')(final_rep)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out = Dense(1, activation="sigmoid")(final_rep)
        model = Model(inputs=[seq1, seq2], outputs=out)
        return model
    
    def cnn_layer(self,input_x,sequence_length,embedding_dim,num_filters=150,filter_sizes= [3,4,5]):
        reshape = Reshape((sequence_length,embedding_dim,1))(input_x)
        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

        maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        return flatten