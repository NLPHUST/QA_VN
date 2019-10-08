import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Dot,Conv2D,Flatten,MaxPool2D
from keras.optimizers import Adam
from layers import Match,MatchTensor
from layers.SpatialGRU import *

class LSTM_MATCH():
    def __init__(self,config):
        self.config = config
        self.model = self.build()
    
    def build(self):
        seq1 = Input(name='seq1', shape=[self.config['seq1_maxlen']])
        seq2 = Input(name='seq2', shape=[self.config['seq2_maxlen']])
        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], trainable = self.config['embed_trainable'])

        seq1_embed = embedding(seq1)
        seq2_embed = embedding(seq2)

        lstm = Bidirectional(LSTM(self.config['hidden_size'],dropout=self.config['dropout_rate']))
        lstm2 = Bidirectional(LSTM(self.config['hidden_size'],dropout=self.config['dropout_rate']))
        seq1_rep = lstm(seq1_embed)
        seq2_rep = lstm(seq2_embed)

        final_rep = concatenate([seq1_rep,seq2_rep])
        final_rep = BatchNormalization()(final_rep)
        final_rep = Dropout(rate=self.config['dropout_rate'])(final_rep)

        if self.config['target_mode'] == 'classification':
            out = Dense(2, activation='softmax')(final_rep)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out = Dense(1,activation ="sigmoid")(final_rep)

        model = Model(inputs=[seq1, seq2], outputs=out)
        return model