import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Dot,Conv2D,Flatten,MaxPool2D
from keras.optimizers import Adam
from layers import Match,MatchTensor
from layers.SpatialGRU import *

class MVRNN():
    def __init__(self,config):
        self.config = config
        self.model = self.build()
    
    def build(self):
        seq1 = Input(name='seq1', shape=[self.config['seq1_maxlen']])
        seq2 = Input(name='seq2', shape=[self.config['seq2_maxlen']])
        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'],weights=[self.config['embed']], trainable = self.config['embed_trainable'])

        seq1_embed = embedding(seq1)
        seq1_embed = Dropout(rate = self.config['dropout_rate'])(seq1_embed)
        seq2_embed = embedding(seq2)
        seq2_embed = Dropout(rate = self.config['dropout_rate'])(seq2_embed)

        lstm1 = Bidirectional(LSTM(self.config['hidden_size'], return_sequences=True))
        lstm2 = Bidirectional(LSTM(self.config['hidden_size'], return_sequences=True))
        seq1_rep = lstm1(seq1_embed)
        seq2_rep = lstm2(seq2_embed)

        cross = Match(match_type='concat')([seq1_rep, seq2_rep])
        cross_reshape = Reshape((-1, ))(cross)

        mm_k = Lambda(lambda x: K.tf.nn.top_k(x, k=100, sorted=True)[0])(cross_reshape)

        pool_flat_drop = Dropout(rate=self.config['dropout_rate'])(mm_k)

        if self.config['target_mode'] == 'classification':
            out = Dense(2, activation='softmax')(pool_flat_drop)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out = Dense(1, activation="sigmoid")(pool_flat_drop)

        model = Model(inputs=[seq1, seq2], outputs=out)
        return model