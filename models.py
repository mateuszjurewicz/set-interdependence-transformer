"""
This file contains all model architectures, both custom and pre-existing.
Links to papers and original implementations are provided, where appropriate.
Additionally, some helper functions related to models can be found here,
particularly for model training, model config reading and validation.

Configuration files for models, per experiment, can be found in the
`model_configs` directory, in the form of json files.
"""
import itertools
import logging
import math
import time
import warnings

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from tqdm import tqdm
from transformers import BertModel, AutoModelForPreTraining

warnings.filterwarnings("ignore", category=UserWarning)


###############################################################################
# Custom Set Attention (2021)
# Allowing for attention between set elements and perm-invar set composition.
###############################################################################


class SetInterDependenceTransformer(nn.Module):
    """
    Set Interdependence Transformer module for attending between set elements
    and its permutation-invariant representation.
    """

    def __init__(self,
                 set_att_in_dim,
                 set_att_out_dim,
                 set_att_n_heads,
                 set_att_n_layers,
                 set_att_n_seeds):
        super(SetInterDependenceTransformer, self).__init__()

        # params
        self.set_att_in_dim = set_att_in_dim
        self.set_att_out_dim = set_att_out_dim
        self.set_att_n_heads = set_att_n_heads
        self.set_att_n_layers = set_att_n_layers
        self.set_att_n_seeds = set_att_n_seeds

        # layers
        self.set_attention_layers = nn.ModuleList(
            [nn.Sequential(SAB(self.set_att_out_dim,
                               self.set_att_out_dim,
                               self.set_att_n_heads, False))
             for _ in range(self.set_att_n_layers - 1)])
        self.set_pooling = PMA(self.set_att_out_dim, self.set_att_n_heads,
                               self.set_att_n_seeds)

    def forward(self, S, E):
        # concat
        Z = torch.cat([S, E], dim=1)

        # attend layers
        for layer in self.set_attention_layers:
            Z = layer(Z)

        # get new S
        S = self.set_pooling(Z)

        # extract new E, preventing cuda mistakes
        indices = torch.tensor([e + 1 for e in range(Z.size()[1] - 1)])
        if torch.cuda.is_available() and next(self.parameters()).is_cuda:
            indices = torch.tensor([e + 1 for e in range(
                Z.size()[1] - 1)]).cuda()
        E = torch.index_select(Z, dim=1, index=indices)

        return S, E


class CustomSetEncoder(nn.Module):
    """
    A block of possibly multiple applications of some form of
    joint set element and set composition attention blocks.
    """

    def __init__(self,
                 set_att_in_dim,
                 set_att_type,
                 set_att_out_dim,
                 set_att_n_heads,
                 set_att_n_layers,
                 set_att_n_blocks,
                 set_att_n_seeds):
        super(CustomSetEncoder, self).__init__()

        # params
        self.set_att_type = set_att_type
        self.set_att_in_dim = set_att_in_dim
        self.set_att_out_dim = set_att_out_dim
        self.set_att_n_heads = set_att_n_heads
        self.set_att_n_layers = set_att_n_layers
        self.set_att_n_blocks = set_att_n_blocks
        self.set_att_n_seeds = set_att_n_seeds

        # first layer needs to 1) resize element embedding dims 2) obtain first set representation
        self.elem_resizer = SAB(self.set_att_in_dim, self.set_att_out_dim, self.set_att_n_heads, False)
        self.first_pool = PMA(self.set_att_out_dim, self.set_att_n_heads, self.set_att_n_seeds)

        # blocks of appropriate type
        if self.set_att_type == 'set_interdependence':

            # remaining blocks of app
            self.main_attention_blocks = nn.ModuleList(
                [SetInterDependenceTransformer(
                    self.set_att_out_dim,
                    self.set_att_out_dim,
                    self.set_att_n_heads,
                    self.set_att_n_layers,
                    self.set_att_n_seeds)
                    for _ in range(self.set_att_n_blocks - 1)]
            )
        else:
            raise (Exception('Unimplemented type of custom set attention'))

    def forward(self, X):
        E = self.elem_resizer(X)
        S = self.first_pool(E)
        for block in self.main_attention_blocks:
            S, E = block(S, E)

        # squeeze the middle dimension (if seeds = 1)
        S = S.squeeze(1)
        return {'embedded_set': S, 'embedded_elements': E}


###############################################################################
# Sentence Encoder
# Simple module using a `bert-base-cased` to obtain sentence embeddings
# for a simplified sentence ordering experiment.
###############################################################################


class SentenceEmbedderBertCased(nn.Module):
    """
    A sentence embedding module using pretrained, uncased BERT.
    Outputs an embedding per sentence passed.
    """

    def __init__(self):
        super(SentenceEmbedderBertCased, self).__init__()
        self.language_model = BertModel. \
            from_pretrained('bert-base-cased', output_hidden_states=True)

    def forward(self, tokenized_bert_batch):
        z = self.language_model(**tokenized_bert_batch)
        z = [z[2][i] for i in (-1, -2, -3, -4)]  # last 4 layers, concat, mean
        z = torch.cat(tuple(z), dim=-1)
        z = torch.mean(z, dim=1).squeeze()
        return z

    @staticmethod
    def prepare_rocstory_batch(batch):
        """
        Take a 5-sentence ROCStory split batch, return 5 separate batches that BERT
        will understand.
        """
        per_sentence_batches = [
            {'input_ids': batch['s{}_input_ids'.format(i + 1)],
             'token_type_ids': batch['s{}_token_type_ids'.format(i + 1)],
             'attention_mask': batch['s{}_attention_mask'.format(i + 1)]}
            for i in range(5)]
        return per_sentence_batches


class OfferEmbedderBertCased(nn.Module):
    """
    An offer embedding module using pretrained, danish-bert-botxo.
    Outputs an embedding per sentence passed.
    """

    def __init__(self):
        super(OfferEmbedderBertCased, self).__init__()
        self.language_model = AutoModelForPreTraining. \
            from_pretrained('danish-bert-botxo',
                            from_tf=True,
                            output_hidden_states=True)

    def forward(self, tokenized_bert_batch):
        z = self.language_model(**tokenized_bert_batch)
        z = [z[2][i] for i in (-1, -2, -3, -4)]  # last 4 layers, concat, mean
        z = torch.cat(tuple(z), dim=-1)
        z = torch.mean(z, dim=1)
        return z

    @staticmethod
    def prepare_procat_batch(batch):
        """
        Take a 200-offer PROCAT split batch, return 200 separate batches that the language
        model will understand.
        """
        per_offer_batches = [
            {'input_ids': batch['s{}_input_ids'.format(i + 1)],
             'token_type_ids': batch['s{}_token_type_ids'.format(i + 1)],
             'attention_mask': batch['s{}_attention_mask'.format(i + 1)]}
            for i in range(200)]
        return per_offer_batches

###############################################################################
# Element Encoders (2017 - 2021)
# Modularized NNs for embedding set elements.
###############################################################################


class ElementEncoderFirstLayer(nn.Module):
    """
    A layer capable of taking a batch of set elements in the form of
    single-element vectors (for float sorting), two-element vectors (TSP),
    vectors of dictionary indices (synthetic) and language-model input
    (sentence ordering, PROCAT).
    """

    def __init__(self,
                 elem_dims,
                 out_dims,
                 embedding_by_dict=False,
                 embedding_by_dict_size=None):

        super(ElementEncoderFirstLayer, self).__init__()
        self.elem_dims = elem_dims
        self.out_dims = out_dims

        # first layer must be aware of input element dimensionality
        # and type (e.g. are inputs dictionary indices)
        if embedding_by_dict:
            self.first_layer = nn.Embedding(embedding_by_dict_size,
                                            out_dims)
        # TODO: add handling of language model inputs
        elif False:
            pass
        else:
            self.first_layer = nn.Linear(elem_dims, out_dims)

    def forward(self, X):
        """
        Take a batch of any dimensionality, return a tensor of size:
        (batch, self.out_dims)
        :param X: batch to predict on
        :return: embedding from 1 appropriate layer (batch, self.out_dims)
        """

        Z = self.first_layer(X)
        return Z


class ElementEncoderLinear(nn.Module):
    def __init__(self,
                 elem_dims,
                 out_dims,
                 num_layers,
                 embedding_by_dict=False,
                 embedding_by_dict_size=None):
        super(ElementEncoderLinear, self).__init__()
        self.elem_dims = elem_dims
        self.out_dims = out_dims
        self.num_layers = num_layers
        self.embedding_by_dict = embedding_by_dict
        self.embedding_by_dict_size = embedding_by_dict_size

        # first layer must be aware of input element dimensionality
        # and type (e.g. are inputs dictionary indices)
        self.first_layer = ElementEncoderFirstLayer(self.elem_dims,
                                                    self.out_dims,
                                                    self.embedding_by_dict,
                                                    self.embedding_by_dict_size)

        # remaining layers
        self.second_plus_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(out_dims, out_dims),
                           nn.ReLU()) for i in range(self.num_layers)])

    def forward(self, X):
        Z = self.first_layer(X)
        for layer in self.second_plus_layers:
            Z = layer(Z)
        return Z


class ElementEncoderSetTransformer(nn.Module):
    def __init__(self,
                 elem_dims,
                 out_dims,
                 num_layers,
                 embedding_by_dict=False,
                 embedding_by_dict_size=None,
                 num_heads=8, ln=False):
        super(ElementEncoderSetTransformer, self).__init__()
        self.elem_dims = elem_dims
        self.out_dims = out_dims
        self.num_layers = num_layers
        self.embedding_by_dict = embedding_by_dict
        self.embedding_by_dict_size = embedding_by_dict_size
        self.num_heads = num_heads
        self.ln = ln

        # first layer must be aware of input element dimensionality
        # and type (e.g. are inputs dictionary indices)
        if self.embedding_by_dict:
            self.first_layer = ElementEncoderFirstLayer(self.elem_dims,
                                                        self.out_dims,
                                                        self.embedding_by_dict,
                                                        self.embedding_by_dict_size)
        else:
            self.first_layer = SAB(self.elem_dims,
                                   self.out_dims,
                                   self.num_heads,
                                   ln=self.ln)

        # remaining layers
        self.second_plus_layers = nn.ModuleList(
            [nn.Sequential(SAB(self.out_dims,
                               self.out_dims,
                               self.num_heads,
                               ln=self.ln)) for i in range(self.num_layers)])

    def forward(self, X):
        Z = self.first_layer(X)
        for layer in self.second_plus_layers:
            Z = layer(Z)
        return Z


###############################################################################
# Transformer Positional Encoding
# vide https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# ###############################################################################


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


###############################################################################
# Set Encoders (2015 - 2021)
# Modularized NNs for embedding entire sets in a permutation invariant way.
###############################################################################


class SetEncoderFirstLayer(nn.Module):
    """
    A layer capable of taking a batch of set elements in the form of
    single-element vectors (for float sorting), two-element vectors (TSP),
    vectors of dictionary indices (synthetic) and language-model input
    (sentence ordering, PROCAT). Same for element and set encoders, as those
    are two separate information paths.
    """

    def __init__(self,
                 elem_dims,
                 out_dims,
                 embedding_by_dict=False,
                 embedding_by_dict_size=None):

        super(SetEncoderFirstLayer, self).__init__()
        self.elem_dims = elem_dims
        self.out_dims = out_dims

        # first layer must be aware of input element dimensionality
        # and type (e.g. are inputs dictionary indices)
        if embedding_by_dict:
            self.first_layer = nn.Embedding(embedding_by_dict_size,
                                            out_dims)
        # TODO: add handling of language model inputs
        elif False:
            pass
        else:
            self.first_layer = nn.Linear(elem_dims, out_dims)

    def forward(self, X):
        """
        Take a batch of any dimensionality, return a tensor of size:
        (batch, self.out_dims)
        :param X: batch to predict on
        :return: embedding from 1 appropriate layer (batch, self.out_dims)
        """

        Z = self.first_layer(X)
        return Z


class SetEncoderRNN(nn.Module):
    def __init__(self,
                 elem_embed_dims,
                 set_embed_dim,
                 set_embed_rnn_layers,
                 set_embed_rnn_bidir,
                 set_embed_dropout):
        super(SetEncoderRNN, self).__init__()

        # element embedding
        self.elem_embed_dims = elem_embed_dims
        # set embedding
        self.set_embed_dims = set_embed_dim // 2 if set_embed_rnn_bidir else set_embed_dim
        self.set_embed_rnn_layers = set_embed_rnn_layers * 2 if set_embed_rnn_bidir else set_embed_rnn_layers
        self.set_embed_rnn_bidir = set_embed_rnn_bidir
        self.set_embed_dropout = set_embed_dropout

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

        # # first layer must be aware of input element dimensionality
        # # and type (e.g. are inputs dictionary indices)
        # self.first_layer = SetEncoderFirstLayer(self.elem_dims,
        #                                         self.elem_embed_dims,
        #                                         self.embedding_by_dict,
        #                                         self.embedding_by_dict_size)

        # actual set encoder
        self.fc1 = nn.Linear(self.elem_embed_dims, self.set_embed_dims)
        self.rnn = nn.LSTM(
            self.set_embed_dims,
            self.set_embed_dims,
            self.set_embed_rnn_layers,
            dropout=self.set_embed_dropout,
            bidirectional=self.set_embed_rnn_bidir,
            batch_first=True)
        self.fc2 = nn.Linear(self.set_embed_dims * 2, self.set_embed_dims)

    def forward(self, X):
        # # per-elem embedding
        # Z = self.first_layer(X)

        # adjust sizes
        Z = self.fc1(X)

        # get first hidden state and cell state
        hidden0 = self.init_hidden(Z)

        encoder_outputs, Z = self.rnn(Z, hidden0)

        if self.set_embed_rnn_bidir:
            # last layer's h and c only, concatenated
            Z = (torch.cat((Z[0][-2:][0], Z[0][-2:][1]), dim=-1),
                 torch.cat((Z[1][-2:][0], Z[1][-2:][1]), dim=-1))
        else:
            Z = (Z[0][-1], Z[1][-1])

        # concatenate (we're splitting them in the decoder)
        Z = torch.cat((Z[0], Z[1]), 1)

        # adjust size again
        Z = self.fc2(Z)

        return {'embedded_set': Z, 'encoder_outputs': encoder_outputs}

    def init_hidden(self, embedded_inputs):
        """
        Initiate hidden units
        :param Tensor embedded_inputs: The embedded input of Pointer-Net
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = embedded_inputs.size(0)

        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.set_embed_rnn_layers,
                                                      batch_size,
                                                      self.set_embed_dims)
        c0 = self.c0.unsqueeze(0).unsqueeze(0).repeat(self.set_embed_rnn_layers,
                                                      batch_size,
                                                      self.set_embed_dims)

        return h0, c0


class SetEncoderReadProcessWrite(nn.Module):
    def __init__(self,
                 elem_dims,
                 elem_embed_dims,
                 elem_embed_num_layers,
                 embedding_by_dict,
                 embedding_by_dict_size,
                 set_embed_dims,
                 set_embed_t_steps):
        super(SetEncoderReadProcessWrite, self).__init__()

        # element embedding
        self.elem_dims = elem_dims
        self.elem_embed_dims = elem_embed_dims
        self.elem_embed_num_layers = elem_embed_num_layers
        self.embedding_by_dict = embedding_by_dict
        self.embedding_by_dict_size = embedding_by_dict_size

        # set embedding
        self.set_embed_dims = set_embed_dims
        self.set_embed_t_steps = set_embed_t_steps

        # # first layer must be aware of input element dimensionality
        # # and type (e.g. are inputs dictionary indices)
        # self.first_layer = SetEncoderFirstLayer(self.elem_dims,
        #                                         self.elem_embed_dims,
        #                                         self.embedding_by_dict,
        #                                         self.embedding_by_dict_size)

        # actual set encoder
        self.fc1 = nn.Linear(self.elem_embed_dims, self.set_embed_dims)
        self.encoder = RPWEncoder(
            self.set_embed_dims,
            self.set_embed_t_steps
        )
        self.fc2 = nn.Linear(self.set_embed_dims * 2, self.set_embed_dims)

    def forward(self, X):
        # # per-elem embedding
        # Z = self.first_layer(X)

        # adjust sizes
        Z = self.fc1(X)

        # set encoding
        Z = self.encoder(Z)

        # adjust size again
        Z = self.fc2(Z)

        return {'embedded_set': Z}


class SetEncoderDeepSets(nn.Module):
    def __init__(self,
                 elem_dims,
                 elem_embed_dims,
                 elem_embed_num_layers,
                 embedding_by_dict,
                 embedding_by_dict_size,
                 set_pooling_type,
                 set_embed_dims,
                 set_embed_num_layers
                 ):
        super(SetEncoderDeepSets, self).__init__()

        # element embedding
        self.elem_dims = elem_dims
        self.elem_embed_dims = elem_embed_dims
        self.elem_embed_num_layers = elem_embed_num_layers
        self.embedding_by_dict = embedding_by_dict
        self.embedding_by_dict_size = embedding_by_dict_size

        # set embedding
        self.set_embed_dims = set_embed_dims
        self.set_embed_num_layers = set_embed_num_layers
        self.set_pooling_type = set_pooling_type

        # # first layer must be aware of input element dimensionality
        # # and type (e.g. are inputs dictionary indices)
        # self.first_layer = SetEncoderFirstLayer(self.elem_dims,
        #                                         self.elem_embed_dims,
        #                                         self.embedding_by_dict,
        #                                         self.embedding_by_dict_size)

        # remaining per-element layers
        self.remaining_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.elem_embed_dims,
                                     self.elem_embed_dims),
                           nn.ReLU()) for _ in
             range(self.elem_embed_num_layers)])

        # set embedding
        if self.set_pooling_type == 'sum':
            self.set_pooling = torch.sum
        elif self.set_pooling_type == 'max':
            self.set_pooling = torch.max
        else:
            # default mean
            self.set_pooling = torch.mean

        self.set_embed_first_layer = nn.Linear(self.elem_embed_dims,
                                               self.set_embed_dims)
        self.set_embed_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.set_embed_dims,
                                     self.set_embed_dims),
                           nn.ReLU()) for _ in
             range(self.set_embed_num_layers)])

    def forward(self, X):

        # # per-elem embedding
        # Z = self.first_layer(X)

        # remaining layers
        for layer in self.remaining_layers:
            Z = layer(X)

        # set embedding
        Z = self.set_pooling(Z, 1)
        Z = self.set_embed_first_layer(Z)
        for layer in self.set_embed_layers:
            Z = layer(Z)
        return {'embedded_set': Z}


class SetEncoderSetTransformer(nn.Module):
    def __init__(self,
                 elem_dim,
                 elem_embed_dim,
                 elem_embed_n_layers,
                 embedding_by_dict,
                 embedding_by_dict_size,
                 set_embed_num_heads,
                 set_embed_num_seeds,
                 set_embed_dim,
                 set_embed_n_layers
                 ):
        super(SetEncoderSetTransformer, self).__init__()

        # element embedding
        self.elem_dim = elem_dim
        self.elem_embed_dim = elem_embed_dim
        self.elem_embed_n_layers = elem_embed_n_layers
        self.embedding_by_dict = embedding_by_dict
        self.embedding_by_dict_size = embedding_by_dict_size

        # set embedding
        self.set_embed_dim = set_embed_dim
        self.set_embed_n_layers = set_embed_n_layers
        self.set_embed_num_heads = set_embed_num_heads
        self.set_embed_num_seeds = set_embed_num_seeds

        # # first layer must be aware of input element dimensionality
        # # and type (e.g. are inputs dictionary indices)
        # self.first_layer = SetEncoderFirstLayer(self.elem_dim,
        #                                         self.elem_embed_dim,
        #                                         self.embedding_by_dict,
        #                                         self.embedding_by_dict_size)

        # remaining per-element layers
        self.remaining_layers = nn.ModuleList(
            [nn.Sequential(SAB(self.elem_embed_dim,
                               self.elem_embed_dim,
                               self.set_embed_num_heads,
                               ln=False)) for _ in
             range(self.elem_embed_n_layers)])

        # set embedding
        self.set_pooling = PMA(self.elem_embed_dim, self.set_embed_num_heads,
                               self.set_embed_num_seeds)
        self.set_embed_first_layer = SAB(self.elem_embed_dim,
                                         self.set_embed_dim,
                                         self.set_embed_num_heads)
        self.set_embed_layers = nn.ModuleList(
            [nn.Sequential(SAB(self.set_embed_dim,
                               self.set_embed_dim,
                               self.set_embed_num_heads)) for _ in
             range(self.set_embed_n_layers)])

    def forward(self, X):

        # # per-elem embedding
        # Z = self.first_layer(X)

        for layer in self.remaining_layers:
            Z = layer(X)

        # set embedding
        Z = self.set_pooling(Z)
        Z = self.set_embed_first_layer(Z)
        for layer in self.set_embed_layers:
            Z = layer(Z)

        # in set transformer, squeeze here
        Z = Z.squeeze(1)
        return {'embedded_set': Z}


###############################################################################
# Permutation Modules
# Add-ons for using a representation of elements in some arbitrary order
# and the (usually) permutation-invariant representation of the set
# to predict a permutation
###############################################################################


class SetToSequencePointerAttention(nn.Module):
    """
    Attention mechanism for a Pointer Net. Implementation follows:
    https://github.com/shirgur/PointerNet/blob/master/PointerNet.py
    """

    def __init__(self, input_dim,
                 hidden_dim):
        """
        Initiate Attention
        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(SetToSequencePointerAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]),
                              requires_grad=False)
        self.soft = torch.nn.Softmax(dim=1)

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass
        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """
        # input: (batch, hidden)
        # context: (batch, seq_len, hidden)

        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1,
                                                           context.size(1))

        # context: (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)

        # ctx: (batch, hidden_dim, seq_len)
        ctx = self.context_linear(context)

        # V: (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # att: (batch, seq_len)
        att = torch.bmm(V, torch.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]

        # alpha: (batch, seq_len)
        alpha = self.soft(att)

        # hidden_state: (batch, hidden)
        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class PermutationInvariantPointerDecoder(nn.Module):
    """
    Decoder adjusted to handling a permutation-invariant hidden state
    and using permutation-equivariant element embeddings as context.
    Should return the same output permutation for any random permutation
    of the input set. Inspiried by the RPW model and its code implementations.
    """

    def __init__(self,
                 elem_embed_dim,
                 set_embed_dim,
                 hidden_dim,
                 masking=True):
        """
        Initiate Decoder
        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(PermutationInvariantPointerDecoder, self).__init__()
        self.elem_embed_dim = elem_embed_dim
        self.set_embed_dim = set_embed_dim
        self.hidden_dim = hidden_dim
        self.masking = masking

        self.context_resizer = nn.Linear(elem_embed_dim, hidden_dim)
        self.set_resizer_hidden = nn.Linear(set_embed_dim // 2, hidden_dim)
        self.set_resizer_cellstate = nn.Linear(set_embed_dim // 2, hidden_dim)
        self.input_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = SetToSequencePointerAttention(hidden_dim, hidden_dim)

        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                decoder_input,
                hidden,
                context=None):
        """
        Decoder - Forward-pass
        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: set representation
        :param Tensor context: RPW-style decoder doesn't take one, here
                               for interface consistency
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.att.init_inf(mask.size())

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        # resize context and set
        embedded_inputs = self.context_resizer(embedded_inputs)
        hidden = self.set_resizer_hidden(hidden[0]), self.set_resizer_cellstate(
            hidden[1])

        def step(x, hidden, embedded_inputs_as_context):
            """
            Recurrence step function
            :param Tensor x: Input at time t
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :param embedded_inputs_as_context -> replaced encoder hidden states
            :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
            """

            # Regular LSTM
            h, c = hidden

            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
            input, forget, cell, out = gates.chunk(4, 1)

            # input, forget, cell, out: (batch, hidden)
            input = torch.sigmoid(input)
            forget = torch.sigmoid(forget)
            cell = torch.tanh(cell)
            out = torch.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * torch.tanh(c_t)

            # Attention section
            hidden_t, output = self.att(h_t,
                                        embedded_inputs_as_context,
                                        torch.eq(mask, 0))
            hidden_t = torch.tanh(
                self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        # Recurrence loop
        for _ in range(input_length):
            h_t, c_t, outs = step(decoder_input, hidden, embedded_inputs)
            hidden = (h_t, c_t)

            # Masking selected inputs
            masked_outs = outs * mask

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1,
                                                                      outs.size()[
                                                                          1])).float()

            # Update mask to ignore seen indices, if masking is enabled
            if self.masking:
                mask = mask * (1 - one_hot_pointers)

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1,
                                                                  self.hidden_dim).byte()

            # Below line aims to fix:
            # UserWarning: indexing with dtype torch.uint8 is now deprecated,
            # please use a dtype torch.bool instead.
            embedding_mask = embedding_mask.bool()

            decoder_input = embedded_inputs[embedding_mask.data].view(
                batch_size, self.hidden_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden


class PermutationSensitivePointerDecoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self,
                 elem_embed_dim,
                 set_embed_dim,
                 hidden_dim,
                 masking=True):
        """
        Initiate Decoder
        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(PermutationSensitivePointerDecoder, self).__init__()
        self.elem_embed_dim = elem_embed_dim
        self.set_embed_dim = set_embed_dim
        self.hidden_dim = hidden_dim
        self.masking = masking

        self.emb_inputs_resizer = nn.Linear(elem_embed_dim, hidden_dim)
        self.set_resizer_hidden = nn.Linear(set_embed_dim // 2, hidden_dim)
        self.set_resizer_cellstate = nn.Linear(set_embed_dim // 2, hidden_dim)
        self.input_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = SetToSequencePointerAttention(hidden_dim, hidden_dim)

        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                decoder_input,
                hidden,
                context):
        """
        Decoder - Forward-pass
        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs or sth else
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.att.init_inf(mask.size())

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        # resize context and set
        embedded_inputs = self.emb_inputs_resizer(embedded_inputs)
        hidden = self.set_resizer_hidden(hidden[0]), self.set_resizer_cellstate(
            hidden[1])

        def step(x, hidden, context):
            """
            Recurrence step function
            :param Tensor x: Input at time t
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
            """
            # Regular LSTM
            # x: (batch, embedding)
            # hidden: ((batch, hidden),
            #          (batch, hidden))
            h, c = hidden

            # gates: (batch, hidden * 4)
            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)

            # input, forget, cell, out: (batch, hidden)
            input, forget, cell, out = gates.chunk(4, 1)

            input = torch.sigmoid(input)
            forget = torch.sigmoid(forget)
            cell = torch.tanh(cell)
            out = torch.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * torch.tanh(c_t)

            # Attention section
            # h_t: (batch, hidden)
            # context: (batch, seq_len, hidden)
            # mask: (batch, seq_len)
            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))
            hidden_t = torch.tanh(
                self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        # Recurrence loop
        for _ in range(input_length):
            # decoder_input: (batch, embedding)
            # hidden: ((batch, hidden),
            #          (batch, hidden))
            h_t, c_t, outs = step(decoder_input, hidden, context)
            hidden = (h_t, c_t)

            # Masking selected inputs
            masked_outs = outs * mask

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1,
                                                                      outs.size()[
                                                                          1])).float()

            # Update mask to ignore seen indices, if masking is enabled
            if self.masking:
                mask = mask * (1 - one_hot_pointers)

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1,
                                                                  self.hidden_dim).byte()

            # Below line aims to fix:
            # UserWarning: indexing with dtype torch.uint8 is now deprecated,
            # please use a dtype torch.bool instead.
            embedding_mask = embedding_mask.bool()

            decoder_input = embedded_inputs[embedding_mask.data].view(
                batch_size, self.hidden_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden


class FutureHistoryBeam(object):
    def __init__(self, beam_size):
        self.beam_size = beam_size

        self.candidates = []
        self.scores = []

    def step(self, prob, prev_beam, f_done):
        pre_score = prob.new_tensor(prev_beam.scores)

        score = prob + pre_score.unsqueeze(-1).expand_as(prob)
        if score.numel() < self.beam_size:
            nbest_score, nbest_ix = score.view(-1).topk(score.numel(),
                                                        largest=False)
        else:
            nbest_score, nbest_ix = score.view(-1).topk(self.beam_size,
                                                        largest=False)

        beam_ix = nbest_ix / prob.size(1)
        token_ix = nbest_ix - beam_ix * prob.size(1)

        done_list, remain_list = [], []
        prev_candidates = prev_beam.candidates
        for b_score, b_ix, t_ix in itertools.zip_longest(nbest_score.tolist(),
                                                         beam_ix.tolist(),
                                                         token_ix.tolist()):
            candidate = prev_candidates[b_ix] + [t_ix]

            if f_done(candidate):
                done_list.append([candidate, b_score])
            else:
                remain_list.append(b_ix)
                self.candidates.append(candidate)
                self.scores.append(b_score)
        return done_list, remain_list


class FutureHistoryPointerDecoder(nn.Module):
    """
    Follows code reimplementations at
    https://github.com/DeepLearnXMU/Pairwise.git
    https://github.com/middlekisser/Pairwise
    Based on the Enhancing Pointer Network for Sentence Ordering
    with Pairwise Ordering Predictions 2020 paper.
    """

    def __init__(self,
                 elem_embedding_dim,
                 set_encoder_dim,
                 hidden_dim,
                 label_dim,
                 pair_dim,
                 lamb_rela,
                 dropout,
                 masking=True):
        super(FutureHistoryPointerDecoder, self).__init__()

        # params
        self.elem_embedding_dim = elem_embedding_dim
        self.set_embedding_dim = set_encoder_dim
        self.emb_dp = dropout
        self.model_dp = dropout
        self.lamb = lamb_rela
        self.critic = None
        labelemb_dim = label_dim
        d_pair = pair_dim

        if not masking:
            raise Exception(
                "Future+History Ptr Net doesn't support no masking.")

        # resizing
        self.element_resizer = nn.Linear(self.elem_embedding_dim, hidden_dim * 2)
        self.set_resizer1 = nn.Linear(self.set_embedding_dim // 2, 2 * hidden_dim)
        self.set_resizer2 = nn.Linear(self.set_embedding_dim // 2, 2 * hidden_dim)

        # lstm decoder
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim * 2, batch_first=True)

        # pointer net
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim * 2, False),
             nn.Linear(hidden_dim * 2, hidden_dim * 2, False),
             nn.Linear(hidden_dim * 2, 1, False)])

        # future ffn
        self.future = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2, False), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, d_pair, False), nn.ReLU(),
            nn.Dropout(0.1))
        self.w3 = nn.Linear(d_pair, 2, False)
        self.hist_left1 = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2, False), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim, False), nn.ReLU(),
            nn.Dropout(0.1))
        # for sind, l2 half dim
        self.hist_left2 = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2, False), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, d_pair, False), nn.ReLU(),
            nn.Dropout(0.1))
        self.wleft1 = nn.Linear(d_pair, 2, False)
        self.wleft2 = nn.Linear(d_pair, 2, False)

        # new key
        d_pair_posi = d_pair + labelemb_dim
        self.pw_k = nn.Linear(d_pair_posi * 4, hidden_dim * 2, False)

        self.pw_e = nn.Linear(hidden_dim, 1, False)

    def equip(self, critic):
        self.critic = critic

    def resize(self, embedded_elements, set_encoding):
        """
        A resizing function, must be called in ALL (both?) versions of the forward pass
        (infer and train).
        :param embedded_elements: embeded elements as a batch
        :param set_encoding: encoded set as a tuple of tensors, as batch
        :return: resized versions of both tensors
        """

        X = self.element_resizer(embedded_elements)
        S1 = self.set_resizer1(set_encoding[0])
        S2 = self.set_resizer2(set_encoding[1])

        return X, (S1, S2)

    def encode_history(self, paragraph, g):
        batch, num, hdim = paragraph.size()

        # B N 1 H
        para_unq2 = paragraph.unsqueeze(2).expand(batch, num, num, hdim)
        # B 1 N H
        para_unq1 = paragraph.unsqueeze(1).expand(batch, num, num, hdim)

        input = torch.cat((para_unq2, para_unq1), -1)
        rela_left1 = self.hist_left1(input)
        rela_left2 = self.hist_left2(input)
        return rela_left1, rela_left2

    def rela_encode(self, paragraph, g):
        batch, num, hdim = paragraph.size()
        # B N 1 H
        para_unq2 = paragraph.unsqueeze(2).expand(batch, num, num, hdim)
        # B 1 N H
        para_unq1 = paragraph.unsqueeze(1).expand(batch, num, num, hdim)
        # B N N H
        input = torch.cat((para_unq2, para_unq1), -1)
        return self.future(input)

    def rela_pred(self, paragraph, g):
        rela_vec = self.rela_encode(paragraph, g)
        rela_p = F.softmax(self.w3(rela_vec), -1)
        rela_vec_diret = torch.cat((rela_vec, rela_p), -1)

        hist_left1, hist_left2 = self.encode_history(paragraph, g)
        left1_p = F.softmax(self.wleft1(hist_left1), -1)
        left2_p = F.softmax(self.wleft2(hist_left2), -1)

        hist_vec_left1 = torch.cat((hist_left1, left1_p), -1)
        hist_vec_left2 = torch.cat((hist_left2, left2_p), -1)

        # prob, label = torch.topk(rela_p, 1)
        return (left1_p, left2_p,
                rela_p), rela_vec_diret, hist_vec_left1, hist_vec_left2

    def key(self, paragraph, rela_vec):
        rela_mask = rela_vec.new_ones(rela_vec.size(0), rela_vec.size(1),
                                      rela_vec.size(2)) \
                    - torch.eye(rela_vec.size(1)).cuda().unsqueeze(0)

        rela_vec_mean = torch.sum(rela_vec * rela_mask.unsqueeze(3),
                                  2) / rela_mask.sum(2, True)
        pre_key = torch.cat((paragraph, rela_vec_mean), -1)
        key = self.linears[1](pre_key)
        return key

    def forward_train(self, x, set_embedding, y):

        # replacing their per-elem and set-repr encoding with own
        batch_size = x.size(0)
        input_length = x.size(1)

        # resize
        x, set_embedding = self.resize(x, set_embedding)

        # replacing old encode function
        document_matrix = x
        hcn = (set_embedding[0].unsqueeze(0), set_embedding[1].unsqueeze(0))

        # obtain tgt len (always == cardinality)
        tgt_len = torch.tensor([input_length])
        tgt_len = tgt_len.repeat(batch_size)

        target = y

        tgt_len_less = tgt_len
        target_less = target

        target_mask = torch.zeros_like(target_less).byte()
        pointed_mask_by_target = torch.zeros_like(target).byte()

        # *************
        # relative order loss
        rela_vec = self.rela_encode(document_matrix, hcn[0])
        score = self.w3(rela_vec)

        # B N N 2
        logp_rela = F.log_softmax(score, -1)

        truth = torch.tril(
            logp_rela.new_ones(input_length, input_length)).long().unsqueeze(
            0).expand(batch_size, input_length, input_length)

        logp_rela = logp_rela[torch.arange(batch_size).unsqueeze(1), target]
        logp_rela = logp_rela[
            torch.arange(batch_size).unsqueeze(1).unsqueeze(2),
            torch.arange(input_length).unsqueeze(0).unsqueeze(
                2), target.unsqueeze(1)]
        loss_rela = self.critic(logp_rela.view(-1, 2),
                                truth.contiguous().view(-1))

        # history loss
        rela_hist_left1, rela_hist_left2 = self.encode_history(document_matrix,
                                                               hcn[0])
        score_left1 = self.wleft1(rela_hist_left1)
        score_left2 = self.wleft2(rela_hist_left2)

        logp_left1 = F.log_softmax(score_left1, -1)
        logp_left2 = F.log_softmax(score_left2, -1)

        logp_left1 = logp_left1[torch.arange(batch_size).unsqueeze(1), target]
        logp_left1 = logp_left1[
            torch.arange(batch_size).unsqueeze(1).unsqueeze(2),
            torch.arange(input_length).unsqueeze(0).unsqueeze(
                2), target.unsqueeze(1)]

        logp_left2 = logp_left2[torch.arange(batch_size).unsqueeze(1), target]
        logp_left2 = logp_left2[
            torch.arange(batch_size).unsqueeze(1).unsqueeze(2),
            torch.arange(input_length).unsqueeze(0).unsqueeze(
                2), target.unsqueeze(1)]

        loss_left1_mask = torch.tril(
            target.new_ones(input_length, input_length), -1).unsqueeze(
            0).expand(batch_size, input_length, input_length)
        truth_left1 = loss_left1_mask - torch.tril(
            target.new_ones(input_length, input_length), -2).unsqueeze(0)

        loss_left2_mask = torch.tril(
            target.new_ones(input_length, input_length), -2).unsqueeze(
            0).expand(batch_size, input_length, input_length)
        truth_left2 = loss_left2_mask - torch.tril(
            target.new_ones(input_length, input_length), -3).unsqueeze(0)

        loss_left1 = self.critic(logp_left1.view(-1, 2),
                                 truth_left1.contiguous().view(-1))
        loss_left2 = self.critic(logp_left2.view(-1, 2),
                                 truth_left2.contiguous().view(-1))

        eye_mask = torch.eye(input_length).byte().unsqueeze(0)

        # if cude available
        if torch.cuda.is_available():
            eye_mask = eye_mask.to(device='cuda')

        rela_mask = torch.ones_like(truth_left1).byte() - eye_mask
        left1_mask = loss_left1_mask.clone()
        left2_mask = loss_left1_mask.clone()

        for b in range(batch_size):
            pointed_mask_by_target[b, :tgt_len[b]] = 1
            target_mask[b, :tgt_len_less[b]] = 1

            rela_mask[b, tgt_len[b]:] = 0
            rela_mask[b, :, tgt_len[b]:] = 0

            left1_mask[b, tgt_len[b]:] = 0
            left2_mask[b, tgt_len[b]:] = 0

            if tgt_len[b] >= 4:
                for ix in range(3, tgt_len[b]):
                    weight = document_matrix.new_ones(ix)
                    weight[-1] = 0
                    negix = torch.multinomial(weight, ix - 1 - 1)
                    left1_mask[b, ix, negix] = 0

                    weight[-1] = 1
                    weight[-2] = 0
                    negix = torch.multinomial(weight, ix - 1 - 1)
                    left2_mask[b, ix, negix] = 0

        loss_rela.masked_fill_(rela_mask.view(-1) == 0, 0)
        loss_rela = loss_rela.view(batch_size, input_length, -1).sum(
            2) / target_mask.sum(1, True).float()
        loss_rela = loss_rela.sum() / batch_size

        loss_left1.masked_fill_(left1_mask.view(-1) == 0, 0)
        loss_left1 = loss_left1.view(batch_size, input_length, -1).sum(
            2) / target_mask.sum(1, True).float()
        loss_left1 = loss_left1.sum() / batch_size

        loss_left2.masked_fill_(left2_mask.view(-1) == 0, 0)
        loss_left2 = loss_left2.view(batch_size, input_length, -1).sum(
            2) / target_mask.sum(1, True).float()
        loss_left2 = loss_left2.sum() / batch_size

        # *************

        # B N-2 H
        dec_inputs = document_matrix[
            torch.arange(document_matrix.size(0)).unsqueeze(1), target_less[:,
                                                                :-1]]
        start = dec_inputs.new_zeros(batch_size, 1, dec_inputs.size(2))
        # B N-1 H
        dec_inputs = torch.cat((start, dec_inputs), 1)

        p_direc = F.softmax(score, -1)
        rela_vec_diret = torch.cat((rela_vec, p_direc), -1)
        p_left1 = F.softmax(score_left1, -1)
        p_left2 = F.softmax(score_left2, -1)

        hist_vec_left1 = torch.cat((rela_hist_left1, p_left1), -1)
        hist_vec_left2 = torch.cat((rela_hist_left2, p_left2), -1)

        dec_outputs = []
        pw_keys = []
        # mask already pointed nodes
        pointed_mask = [rela_mask.new_zeros(batch_size, 1, input_length)]

        eye_zeros = torch.ones_like(eye_mask) - eye_mask
        eye_zeros = eye_zeros.unsqueeze(-1)

        for t in range(input_length):
            if t == 0:
                rela_mask = rela_mask.unsqueeze(-1)
                l1_mask = torch.zeros_like(rela_mask)
                l2_mask = torch.zeros_like(rela_mask)
            else:
                # B (left1)
                tar = target[:, t - 1]

                # future
                rela_mask[torch.arange(batch_size), tar] = 0
                rela_mask[torch.arange(batch_size), :, tar] = 0

                l1_mask = torch.zeros_like(rela_mask)
                l2_mask = torch.zeros_like(rela_mask)

                l1_mask[torch.arange(batch_size), :, tar] = 1
                if t > 1:
                    l2_mask[torch.arange(batch_size), :, target[:, t - 2]] = 1

                pm = pointed_mask[-1].clone().detach()
                pm[torch.arange(batch_size), :, tar] = 1
                pointed_mask.append(pm)

            # history information
            cur_hist_l1 = hist_vec_left1.masked_fill(l1_mask == 0, 0).sum(2)
            cur_hist_l2 = hist_vec_left2.masked_fill(l2_mask == 0, 0).sum(2)

            # future information
            rela_vec_diret.masked_fill_(rela_mask == 0, 0)
            forw_pw = rela_vec_diret.mean(2)
            back_pw = rela_vec_diret.mean(1)

            pw_info = torch.cat((cur_hist_l1, cur_hist_l2, forw_pw, back_pw),
                                -1)
            pw_key = self.pw_k(pw_info)
            pw_keys.append(pw_key.unsqueeze(1))

            dec_inp = dec_inputs[:, t:t + 1]

            # B 1 H
            output, hcn = self.decoder(dec_inp, hcn)
            dec_outputs.append(output)

        # B N-1 H
        dec_outputs = torch.cat(dec_outputs, 1)
        # B N-1 1 H
        query = self.linears[0](dec_outputs).unsqueeze(2)

        key = torch.cat(pw_keys, 1)
        # B N-1 N H
        e = torch.tanh(query + key)
        # B N-1 N
        e = self.linears[2](e).squeeze(-1)

        # B N-1 N
        pointed_mask = torch.cat(pointed_mask, 1)
        pointed_mask_by_target = pointed_mask_by_target.unsqueeze(1).expand_as(
            pointed_mask)

        e.masked_fill_(pointed_mask == 1, -1e9)
        e.masked_fill_(pointed_mask_by_target == 0, -1e9)

        logp = F.log_softmax(e, dim=-1)
        logp = logp.view(-1, logp.size(-1))
        loss = self.critic(logp, target_less.contiguous().view(-1))

        loss.masked_fill_(target_mask.view(-1) == 0, 0)
        loss = loss.sum() / batch_size

        total_loss = loss + (loss_rela + loss_left1 + loss_left2) * self.lamb
        if torch.isnan(total_loss):
            exit('nan')
        return total_loss, (loss, loss_rela * self.lamb, loss_left1 * self.lamb,
                            loss_left2 * self.lamb)

    def rela_att(self, prev_h, rela, rela_k, rela_mask):
        # B 1 H
        q = self.rela_q(prev_h).transpose(0, 1)
        e = self.rela_e(torch.tanh(q + rela_k))

        e.masked_fill_(rela_mask == 0, -1e9)
        alpha = F.softmax(e, 1)
        context = torch.sum(alpha * rela, 1, True)
        return context

    def stepv2(self, prev_y, prev_handc, keys, mask, rela_vec, hist_left1,
               hist_left2, rela_mask, l1_mask, l2_mask):
        '''
        :param prev_y: (seq_len=B, 1, H)
        :param prev_handc: (1, B, H)
        :return:
        '''

        _, (h, c) = self.decoder(prev_y, prev_handc)
        # 1 B H-> B H-> B 1 H
        query = h.squeeze(0).unsqueeze(1)
        query = self.linears[0](query)

        # history
        left1 = hist_left1.masked_fill(l1_mask.unsqueeze(-1) == 0, 0).sum(2)
        left2 = hist_left2.masked_fill(l2_mask.unsqueeze(-1) == 0, 0).sum(2)

        # future
        rela_vec.masked_fill_(rela_mask.unsqueeze(-1) == 0, 0)
        forw_futu = rela_vec.mean(2)
        back_futu = rela_vec.mean(1)

        pw = torch.cat((left1, left2, forw_futu, back_futu), -1)
        keys = self.pw_k(pw)

        # B N H
        e = torch.tanh(query + keys)
        # B N
        e = self.linears[2](e).squeeze(2)
        e.masked_fill_(mask, -1e9)

        logp = F.log_softmax(e, dim=-1)
        return h, c, logp

    def load_pretrained_emb(self, emb):
        self.src_embed = nn.Embedding.from_pretrained(emb, freeze=False)

    def forward(self, X, decoder_input0, set_embedding, ctx, beam_size=1):
        """
        Perform inference on a batch of x, return actual pointers.
        :param X - embedded elements
        :param decoder_input0 - not used, kept for interface
        :param set_embedding - embedding of the set as a whole
        :param ctx - not used, kept for interface
        :param beam_size - kept at 1
        """

        predictions = []

        # resize
        X, set_embedding = self.resize(X, set_embedding)

        for i, x in enumerate(X):
            # get x as 1-elem batch
            x = x.unsqueeze(0)

            # get set embedding tuple
            s = set_embedding[0][i].unsqueeze(0), set_embedding[1][i].unsqueeze(0)

            # infer on single example
            _, p = self.infer_single(x, s)
            predictions.append(p.squeeze(0))

        y_hat = torch.stack(predictions)

        # adhering to interface
        return (None, y_hat), None

    def infer_single(self, x_single_elem_batch, set_single_elem_batch, beam_size=1):
        """
        Instead of the usual forward, I'm using infer to make predictions
        without tracking all the losses needed for training this model.
        It should be put in eval() mode first, and then back to train()
        afterwards, possibly, as I check whether a model can make a prediction
        prior to training in most training scripts.
        :param x_single_elem_batch: a tensor of size (batch_size, set_size, elem_dims)
        :param set_single_elem_batch: a tuple of 2 tensors (batch_size, set_encoding_dim)
        :return: a predicted tensor of pointers of size (batch_size, set_size)
        """

        sentences = x_single_elem_batch
        dec_init = set_single_elem_batch

        dec_init = (dec_init[0].unsqueeze(1), dec_init[1].unsqueeze(1))

        document = sentences.squeeze(0)
        T, H = document.size()

        keys = self.linears[1](sentences)

        # future
        rela_out, rela_vec, hist_left1, hist_left2 = self.rela_pred(sentences,
                                                                    dec_init[
                                                                        0])

        eye_mask = torch.eye(T).byte()
        eye_zeros = torch.ones_like(eye_mask) - eye_mask

        # if cude available
        if torch.cuda.is_available():
            eye_mask = eye_mask.to(device='cuda')
            eye_zeros = eye_zeros.to(device='cuda')

        W = beam_size

        prev_beam = FutureHistoryBeam(beam_size)
        prev_beam.candidates = [[]]
        prev_beam.scores = [0]

        target_t = T - 1

        f_done = (lambda x: len(x) == target_t)

        valid_size = W
        hyp_list = []

        for t in range(target_t):
            candidates = prev_beam.candidates
            if t == 0:
                # start
                dec_input = sentences.new_zeros(1, 1, H)
                pointed_mask = sentences.new_zeros(1, T).bool()

                rela_mask = eye_zeros.unsqueeze(0)

                l1_mask = torch.zeros_like(rela_mask)
                l2_mask = torch.zeros_like(rela_mask)
            else:
                index = sentences.new_tensor(
                    list(map(lambda cand: cand[-1], candidates))).long()
                # beam 1 H
                dec_input = document[index].unsqueeze(1)

                temp_batch = index.size(0)

                pointed_mask[torch.arange(temp_batch), index] = 1

                rela_mask[torch.arange(temp_batch), :, index] = 0
                rela_mask[torch.arange(temp_batch), index] = 0

                l1_mask = torch.zeros_like(rela_mask)
                l2_mask = torch.zeros_like(rela_mask)

                l1_mask[torch.arange(temp_batch), :, index] = 1
                if t > 1:
                    left2_index = index.new_tensor(
                        list(map(lambda cand: cand[-2], candidates)))
                    l2_mask[torch.arange(temp_batch), :, left2_index] = 1

            dec_h, dec_c, log_prob = self.stepv2(dec_input, dec_init, keys,
                                                 pointed_mask,
                                                 rela_vec, hist_left1,
                                                 hist_left2, rela_mask,
                                                 l1_mask, l2_mask)

            next_beam = FutureHistoryBeam(valid_size)
            done_list, remain_list = next_beam.step(-log_prob, prev_beam,
                                                    f_done)
            hyp_list.extend(done_list)
            valid_size -= len(done_list)

            if valid_size == 0:
                break

            beam_remain_ix = x_single_elem_batch[0].new_tensor(remain_list)
            # needs long here
            beam_remain_ix = beam_remain_ix.long()

            dec_h = dec_h.index_select(1, beam_remain_ix)
            dec_c = dec_c.index_select(1, beam_remain_ix)
            dec_init = (dec_h, dec_c)

            pointed_mask = pointed_mask.index_select(0, beam_remain_ix)

            rela_mask = rela_mask.index_select(0, beam_remain_ix)
            rela_vec = rela_vec.index_select(0, beam_remain_ix)

            hist_left1 = hist_left1.index_select(0, beam_remain_ix)
            hist_left2 = hist_left2.index_select(0, beam_remain_ix)

            prev_beam = next_beam

        score = dec_h.new_tensor([hyp[1] for hyp in hyp_list])
        sort_score, sort_ix = torch.sort(score)
        output = []
        for ix in sort_ix.tolist():
            output.append((hyp_list[ix][0], score[ix].item()))
        best_output = output[0][0]

        the_last = list(set(list(range(T))).difference(set(best_output)))
        best_output.append(the_last[0])

        # switching order here to match other abstractions
        return rela_out, torch.tensor(best_output).unsqueeze(0)


###############################################################################
# Set-to-Sequence (2021)
# Full model capable of learning a mapping from input sets to their permuted
# sequences. Takes advantage of the element encoders and set encoders, in ways
# specified in the json config files.
###############################################################################


class SetToSequence(nn.Module):
    """
    A full set-to-sequence prediction model.
    """

    def __init__(self,
                 elem_dim,
                 elem_embedding_encoder_type,
                 elem_embedding_dim,
                 elem_embedding_by_dict,
                 elem_embedding_by_dict_size,
                 elem_embedding_n_layers,
                 elem_embedding_add_positonal_encoding,
                 set_encoder_type,
                 set_encoder_custom_attention_type,  # only if set_encoder_type == custom
                 set_encoder_custom_attention_n_layers,  # only if set_encoder_type == custom (layers != blocks)
                 set_encoder_rpw_t_steps,  # only RPW
                 set_encoder_rnn_layers,  # only RNN
                 set_encoder_rnn_bidir,  # only RNN
                 set_encoder_rnn_dropout,  # only RNN
                 set_encoder_pooling_type,  # only DeepSets
                 set_encoder_num_heads,  # only SetTransformer
                 set_encoder_num_seeds,  # only SetTransformer
                 set_encoder_dim,
                 set_encoder_n_layers,
                 context_rnn_used,
                 context_rnn_dim,
                 context_rnn_layers,
                 context_rnn_bidir,
                 context_rnn_dropout,
                 permute_module_type,
                 permute_module_is_concat,
                 permute_module_hidden_dim,
                 permute_module_dropout,
                 permute_module_bidir,
                 permute_module_masking,
                 permute_module_label_dim,  # only FutureHistory
                 permute_module_pair_dim,  # only FutureHistory
                 permute_module_lamb_rela,  # only FutureHistory
                 sentence_ordering=False,
                 catalog_ordering=False,
                 ):

        super(SetToSequence, self).__init__()

        # element embedding config
        self.elem_embedding_encoder_type = elem_embedding_encoder_type
        self.elem_dim = elem_dim
        self.elem_embedding_dim = elem_embedding_dim
        self.elem_embedding_by_dict = elem_embedding_by_dict
        self.elem_embedding_by_dict_size = elem_embedding_by_dict_size
        self.elem_embedding_n_layers = elem_embedding_n_layers

        # elem embedding positional encoding
        self.elem_embedding_add_positonal_encoding = elem_embedding_add_positonal_encoding

        # (optional) sentence embedding for language model element embedders
        if sentence_ordering:
            # last 4 layers, concatenated, mean, elem_dims will equal 3072 (set in model config)
            self.sentence_ordering = True
            self.sentence_embedder = SentenceEmbedderBertCased()
        else:
            self.sentence_ordering = False

        # (optional) offer text embedding for language model element embedders
        if catalog_ordering:
            # last 4 layers, concatenated, mean, elem_dims will equal 3072 (set in model config)
            self.catalog_ordering = True
            self.offer_embedder = OfferEmbedderBertCased()
        else:
            self.catalog_ordering = False

        # set embedding config
        self.set_encoder_type = set_encoder_type
        self.set_encoder_rpw_t_steps = set_encoder_rpw_t_steps
        self.set_encoder_rnn_layers = set_encoder_rnn_layers
        self.set_encoder_rnn_bidir = set_encoder_rnn_bidir
        self.set_encoder_rnn_dropout = set_encoder_rnn_dropout
        self.set_encoder_pooling_type = set_encoder_pooling_type
        self.set_encoder_num_heads = set_encoder_num_heads
        self.set_encoder_num_seeds = set_encoder_num_seeds
        self.set_encoder_dim = set_encoder_dim
        self.set_encoder_n_layers = set_encoder_n_layers
        self.set_encoder_custom_attention_type = set_encoder_custom_attention_type
        self.set_encoder_custom_attention_n_layers = set_encoder_custom_attention_n_layers

        # context rnn (optional)
        self.context_rnn_used = context_rnn_used
        self.context_rnn_dim = context_rnn_dim
        self.context_rnn_layers = context_rnn_layers
        self.context_rnn_bidir = context_rnn_bidir
        self.context_rnn_dropout = context_rnn_dropout

        # permutation module config
        self.permute_module_type = permute_module_type
        self.permute_module_is_concat = permute_module_is_concat
        self.permute_module_hidden_dim = permute_module_hidden_dim
        self.permute_module_dropout = permute_module_dropout
        self.permute_module_bidir = permute_module_bidir
        self.permute_module_masking = permute_module_masking
        self.permute_module_label_dim = permute_module_label_dim  # future-history specific
        self.permute_module_pair_dim = permute_module_pair_dim  # future-history specific
        self.permute_module_lamb_rela = permute_module_lamb_rela  # future-history specific

        # element embedding module
        if self.elem_embedding_encoder_type == 'linear':
            self.elem_encoder = ElementEncoderLinear(
                self.elem_dim,
                self.elem_embedding_dim,
                self.elem_embedding_n_layers,
                self.elem_embedding_by_dict,
                self.elem_embedding_by_dict_size)

        elif self.elem_embedding_encoder_type == 'settransformer':
            self.elem_encoder = ElementEncoderSetTransformer(
                self.elem_dim,
                self.elem_embedding_dim,
                self.elem_embedding_n_layers,
                self.elem_embedding_by_dict,
                self.elem_embedding_by_dict_size
            )

        # positional encoding for elem embedding
        if self.elem_embedding_add_positonal_encoding:
            self.elem_positional_encoding = PositionalEncoding(
                self.elem_embedding_dim)

        # set embedding module
        if self.set_encoder_type == 'rnn':
            self.set_embedding = SetEncoderRNN(
                self.elem_embedding_dim,
                self.set_encoder_dim,
                self.set_encoder_rnn_layers,
                self.set_encoder_rnn_bidir,
                self.set_encoder_rnn_dropout)
        elif self.set_encoder_type == 'rpw':
            self.set_embedding = SetEncoderReadProcessWrite(
                self.elem_dim,
                self.elem_embedding_dim,
                self.elem_embedding_n_layers,
                self.elem_embedding_by_dict,
                self.elem_embedding_by_dict_size,
                self.set_encoder_dim,
                self.set_encoder_rpw_t_steps)
        elif self.set_encoder_type == 'deepsets':
            self.set_embedding = SetEncoderDeepSets(
                self.elem_dim,
                self.elem_embedding_dim,
                self.elem_embedding_n_layers,
                self.elem_embedding_by_dict,
                self.elem_embedding_by_dict_size,
                self.set_encoder_pooling_type,
                self.set_encoder_dim,
                self.set_encoder_n_layers)
        elif self.set_encoder_type == 'settransformer':
            self.set_embedding = SetEncoderSetTransformer(
                self.elem_dim,
                self.elem_embedding_dim,
                self.elem_embedding_n_layers,
                self.elem_embedding_by_dict,
                self.elem_embedding_by_dict_size,
                self.set_encoder_num_heads,
                self.set_encoder_num_seeds,
                self.set_encoder_dim,
                self.set_encoder_n_layers)
        elif self.set_encoder_type == 'custom':
            self.set_embedding = CustomSetEncoder(
                self.elem_embedding_dim,
                self.set_encoder_custom_attention_type,
                self.set_encoder_dim,
                self.set_encoder_num_heads,
                self.set_encoder_custom_attention_n_layers,
                self.set_encoder_n_layers,  # actual n blocks
                self.set_encoder_num_seeds
            )

        # RNN for context improved performance on TSP
        if self.context_rnn_used:
            self.context_rnn = SetEncoderRNN(
                self.elem_embedding_dim,
                self.context_rnn_dim,
                self.context_rnn_layers,
                self.context_rnn_bidir,
                self.context_rnn_dropout)

        # permutation module
        if self.permute_module_type == 'permutation_invariant_pointer':
            self.decoder = PermutationInvariantPointerDecoder(
                self.elem_embedding_dim,
                self.set_encoder_dim,
                self.permute_module_hidden_dim,
                self.permute_module_masking)
            self.decoder_input0 = Parameter(torch.FloatTensor(
                self.permute_module_hidden_dim), requires_grad=False)
            # Initialize decoder_input0
            nn.init.uniform_(self.decoder_input0, -1, 1)

        elif self.permute_module_type == 'permutation_sensitive_pointer':
            self.decoder = PermutationSensitivePointerDecoder(
                self.elem_embedding_dim,
                self.set_encoder_dim,
                self.permute_module_hidden_dim,
                self.permute_module_masking)
            self.decoder_input0 = Parameter(torch.FloatTensor(
                self.permute_module_hidden_dim), requires_grad=False)
            # Initialize decoder_input0
            nn.init.uniform_(self.decoder_input0, -1, 1)

        elif self.permute_module_type == 'futurehistory':
            self.decoder = FutureHistoryPointerDecoder(
                self.elem_embedding_dim,
                self.set_encoder_dim,
                self.permute_module_hidden_dim,
                self.permute_module_label_dim,
                self.permute_module_pair_dim,
                self.permute_module_lamb_rela,
                self.permute_module_dropout,
                self.permute_module_masking)

            # not technically used in FH, kept for interface consistency
            self.decoder_input0 = Parameter(torch.FloatTensor(
                self.permute_module_hidden_dim), requires_grad=False)
            # Initialize decoder_input0
            nn.init.uniform_(self.decoder_input0, -1, 1)

    def embed_sentences(self, X):
        """Handle ROCStory input"""
        # first, unpack sentence batches if ROCStory
        s1b, s2b, s3b, s4b, s5b = self.sentence_embedder.prepare_rocstory_batch(X)

        # embed sentences in batches
        s1eb = self.sentence_embedder(s1b)
        s2eb = self.sentence_embedder(s2b)
        s3eb = self.sentence_embedder(s3b)
        s4eb = self.sentence_embedder(s4b)
        s5eb = self.sentence_embedder(s5b)

        # stack into a batch
        embedded_sentences = torch.stack([s1eb, s2eb, s3eb, s4eb, s5eb], dim=1)

        return embedded_sentences

    def embed_offers(self, X):
        """Handle PROCAT input"""
        # first, unpack sentence batches
        nth_offer_batches = self.offer_embedder.prepare_procat_batch(X)

        # embed via language model
        emb_offers = [self.offer_embedder(b) for b in nth_offer_batches]

        # embed and stack into a batch
        embedded_offers = torch.stack(emb_offers, dim=1)

        return embedded_offers

    def forward(self, inputs, target=None):
        """
        PointerNet - Forward-pass
        :param Tensor inputs: Input sequence (batch x seq_len x elem_dim)
        :param Tensor target: (optional, only for training future+history permutation module)
        :return: Pointers probabilities and indices [or losses if future+history and in training mode)
        """
        # (optional) handle sentence input for sentence ordering
        if self.sentence_ordering:
            inputs = self.embed_sentences(inputs)

        # (optional) handle catalog input for structure prediction
        if self.catalog_ordering:
            inputs = self.embed_offers(inputs)

        # inputs: (batch x seq_len x elem_dim)
        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        # element encoding
        if self.elem_embedding_by_dict:
            reshaped_inputs = inputs.long()
        else:
            reshaped_inputs = inputs.float()

        # element encoding
        embedded_inputs = self.elem_encoder(reshaped_inputs).view(batch_size,
                                                                  input_length,
                                                                  -1)
        # positional encoding
        if self.elem_embedding_add_positonal_encoding:
            embedded_inputs = self.elem_positional_encoding(
                embedded_inputs * math.sqrt(self.elem_embedding_dim))

        # set encoding
        set_encoding = self.set_embedding(embedded_inputs)
        embedded_set = set_encoding['embedded_set']

        # (optional) context
        # get encoder outputs as context via rnn
        if self.set_encoder_type == 'rnn':
            context = set_encoding['encoder_outputs']
        elif self.context_rnn_used:
            set_context = self.context_rnn(embedded_inputs)
            context = set_context['encoder_outputs']
        else:
            context = None

        # obtain first decoder hidden state
        decoder_hidden0 = torch.split(embedded_set, self.set_encoder_dim // 2,
                                      dim=1)

        # decoder_input0: (batch,  embedding)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        # pass to a permutation module

        # have to handle training and pure inference differently
        # due to the future+history permutation module possibility
        # if no y (target) has been passed, proceed as normal

        if self.permute_module_type == 'futurehistory' and target is not None:
            loss, (point_loss, rela_loss, left1_loss, left2_loss) = self.decoder.forward_train(
                embedded_inputs,
                decoder_hidden0,
                target)
            return loss, (point_loss, rela_loss, left1_loss, left2_loss)

        # otherwise, proceed normally
        else:
            (outputs, pointers), decoder_hidden = self.decoder(
                embedded_inputs,
                decoder_input0,
                decoder_hidden0,
                context)
            return outputs, pointers


###############################################################################
###############################################################################
# Pointer Network (2015)
# One of the first model architectures applicable to set-to-sequence
# challenges (consist of a permutation-sensitive set encoder and outputs
# a reordering. Involves a modified attention mechanism, works with inputs of
# varying length. Includes an optional masking mechanism to prevent the model
# from pointing to the same element of the input sequence twice.
# Original paper: https://arxiv.org/abs/1506.03134
# Implementation follows:
# https://github.com/shirgur/PointerNet/blob/master/PointerNet.py
###############################################################################
###############################################################################


class PointerEncoder(nn.Module):
    """
    Encoder class for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir):
        """
        Initiate Encoder
        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int n_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerEncoder, self).__init__()
        self.hidden_dim = hidden_dim // 2 if bidir else hidden_dim
        self.n_layers = n_layers * 2 if bidir else n_layers
        self.bidir = bidir
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            n_layers,
                            dropout=dropout,
                            bidirectional=bidir,
                            batch_first=True)

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs, hidden):
        """
        Encoder - Forward-pass
        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor hidden: Initiated hidden units for the LSTMs (h, c)
        :return: LSTMs outputs and hidden units (h, c)
        """

        # if batch_first = True, not needed:
        # embedded_inputs = embedded_inputs.permute(1, 0, 2)
        torch.set_default_dtype(torch.float64)
        outputs, hidden = self.lstm(embedded_inputs, hidden)

        return outputs, hidden

    def init_hidden(self, embedded_inputs):
        """
        Initiate hidden units
        :param Tensor embedded_inputs: The embedded input of Pointer-Net
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = embedded_inputs.size(0)

        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        c0 = self.c0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)

        return h0, c0


class PointerAttention(nn.Module):
    """
    Attention model for Pointer-Net.
    """

    def __init__(self, input_dim,
                 hidden_dim):
        """
        Initiate Attention
        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(PointerAttention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]),
                              requires_grad=False)
        self.soft = torch.nn.Softmax(dim=1)

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass
        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """
        # input: (batch, hidden)
        # context: (batch, seq_len, hidden)

        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1,
                                                           context.size(1))

        # context: (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)

        # ctx: (batch, hidden_dim, seq_len)
        ctx = self.context_linear(context)

        # V: (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # att: (batch, seq_len)
        att = torch.bmm(V, torch.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]

        # alpha: (batch, seq_len)
        alpha = self.soft(att)

        # hidden_state: (batch, hidden)
        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class PointerDecoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim, masking=True,
                 output_length=None):
        """
        Initiate Decoder
        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(PointerDecoder, self).__init__()
        self.masking = masking
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_length = output_length

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = PointerAttention(hidden_dim, hidden_dim)

        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                decoder_input,
                hidden,
                context):
        """
        Decoder - Forward-pass
        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.att.init_inf(mask.size())

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        def step(x, hidden):
            """
            Recurrence step function
            :param Tensor x: Input at time t
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
            """

            # Regular LSTM
            # x: (batch, embedding)
            # hidden: ((batch, hidden),
            #          (batch, hidden))
            h, c = hidden

            # gates: (batch, hidden * 4)
            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)

            # input, forget, cell, out: (batch, hidden)
            input, forget, cell, out = gates.chunk(4, 1)

            input = torch.sigmoid(input)
            forget = torch.sigmoid(forget)
            cell = torch.tanh(cell)
            out = torch.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * torch.tanh(c_t)

            # Attention section
            # h_t: (batch, hidden)
            # context: (batch, seq_len, hidden)
            # mask: (batch, seq_len)
            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))
            hidden_t = torch.tanh(
                self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        # Recurrence loop
        output_length = input_length
        if self.output_length:
            output_length = self.output_length
        for _ in range(output_length):
            # decoder_input: (batch, embedding)
            # hidden: ((batch, hidden),
            #          (batch, hidden))
            h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)

            # Masking selected inputs
            masked_outs = outs * mask

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1,
                                                                      outs.size()[
                                                                          1])).float()

            # Update mask to ignore seen indices, if masking is enabled
            if self.masking:
                mask = mask * (1 - one_hot_pointers)

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1,
                                                                  self.embedding_dim).byte()

            # Below line aims to fixes:
            # UserWarning: indexing with dtype torch.uint8 is now deprecated,
            # please use a dtype torch.bool instead.
            embedding_mask = embedding_mask.bool()

            decoder_input = embedded_inputs[embedding_mask.data].view(
                batch_size, self.embedding_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden


class PointerNetwork(nn.Module):
    """
    Pointer-Net, with optional masking to prevent
    pointing to the same element twice (and never pointing to another).
    """

    def __init__(self,
                 elem_dims,
                 embedding_dim,
                 hidden_dim,
                 lstm_layers,
                 dropout,
                 bidir=False,
                 masking=True,
                 output_length=None,
                 embedding_by_dict=False,
                 embedding_by_dict_size=None):
        """
        Initiate Pointer-Net
        :param int elem_dims: dimensions of each invdividual set element
        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.masking = masking
        self.output_length = output_length
        self.embedding_by_dict = embedding_by_dict
        self.embedding_by_dict_size = embedding_by_dict_size
        if embedding_by_dict:
            self.embedding = nn.Embedding(embedding_by_dict_size,
                                          embedding_dim)
        else:
            self.embedding = nn.Linear(elem_dims, embedding_dim)
        self.encoder = PointerEncoder(embedding_dim,
                                      hidden_dim,
                                      lstm_layers,
                                      dropout,
                                      bidir)
        self.decoder = PointerDecoder(embedding_dim, hidden_dim,
                                      masking=self.masking,
                                      output_length=self.output_length)
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim),
                                        requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        """
        PointerNet - Forward-pass
        :param Tensor inputs: Input sequence (batch x seq_len x elem_dim)
        :return: Pointers probabilities and indices
        """
        # inputs: (batch x seq_len x elem_dim)
        # for TSP in 2D, elem_dim = 2
        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        # decoder_input0: (batch,  embedding)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        # inputs: (batch * seq_len, elem_dim)
        input = inputs.view(batch_size * input_length, -1)

        # embedded_inputs: (batch, seq_len, embedding)
        if self.embedding_by_dict:
            input = input.long()
        else:
            input = input.float()
        embedded_inputs = self.embedding(input).view(batch_size,
                                                     input_length, -1)

        # encoder_hidden0: [(num_lstms, batch_size,  hidden),
        #                   (num_lstms, batch_size,  hidden]
        # where the length depends on number of lstms & bidir
        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs)

        # encoder_outputs: (batch_size, seq_len, hidden)
        # encoder_hidden: [(num_lstms, batch_size, hidden),
        #                  (num_lstms, batch_size, hidden]
        # where the length depends on number of lstms & bidir
        encoder_outputs, encoder_hidden = self.encoder(embedded_inputs,
                                                       encoder_hidden0)

        if self.bidir:
            # last layer's h and c only, concatenated
            decoder_hidden0 = (
                torch.cat(
                    (encoder_hidden[0][-2:][0], encoder_hidden[0][-2:][1]),
                    dim=-1),
                torch.cat(
                    (encoder_hidden[1][-2:][0], encoder_hidden[1][-2:][1]),
                    dim=-1))
        else:
            # decoder_hidden0: ((batch, hidden),
            #                   (batch, hidden))
            decoder_hidden0 = (encoder_hidden[0][-1],
                               encoder_hidden[1][-1])
        (outputs, pointers), decoder_hidden = self.decoder(embedded_inputs,
                                                           decoder_input0,
                                                           decoder_hidden0,
                                                           encoder_outputs)

        return outputs, pointers


###############################################################################
###############################################################################
# Read-Process-and-Write (2017)
# RPW model, an adjusted LSTM with attention resulting in permutation invariant
# representation.
# Original paper: https://research.google/pubs/pub44871/
# Implementation code not provided by the authors.
###############################################################################
###############################################################################


class RPWEncoder(nn.Module):
    """
    Encoder class for Read-Process-and-Write (RPW).
    """

    def __init__(self,
                 hidden_dim,
                 t_steps,
                 n_layers=1):
        """
        Initiate Encoder
        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int t_steps; number of steps of the permutation invariant lstm
        :param int n_layers; hardcoded to be 1 for simplicity for now
        """

        super(RPWEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.t_steps = t_steps
        self.n_layers = n_layers

        # increasing hidden_to_hidden to hidden_dim * 2 for q_star
        self.hidden_to_hidden = nn.Linear(hidden_dim * 2, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs):
        """
        Encoder - Forward-pass
        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :return: LSTMs outputs and hidden units (h, c)
        """
        # memory: (batch, seq_len, hidden)
        # is the embedded inputs
        memory = embedded_inputs

        # intialize q_star (hidden state) and c (cell state)
        # q_star: (batch, embedding * 2)
        # cell_state: (batch, embedding)
        #  no additional dimension since n_lstms = 1 in encode
        q_star, cell_state = self.init_hidden(embedded_inputs)

        def step(q_star, cell_state, memory):
            """
            Recurrence step function
            :param Tensor q_star: query vector, perm-invariant state
            :param Tensor memory: memory vector (embedded input sequence)
            :return: Tensor q_t, after each t step
            """

            ### Part 1 | Modified LSTM

            # removed the self.input_to_hidden,
            # as we're not supposed to have input in an RPW encoder lstm
            # gates: (batch, hidden * 4)
            # consider if q_star isn't enough for both hidden and cell st.
            # you would then not need to adjust the weights of h_t_h by *2
            gates = self.hidden_to_hidden(q_star)

            # input, forget, cell, out: (batch, hidden)
            input, forget, cell, out = gates.chunk(4, 1)

            input = torch.sigmoid(input)
            forget = torch.sigmoid(forget)
            cell = torch.tanh(cell)
            out = torch.sigmoid(out)

            # c_t: (batch, hidden)
            # h_t: (batch, hidden)
            c_t = (forget * cell_state) + (input * cell)
            h_t = out * torch.tanh(c_t)

            ### Part 2 | Attention

            # RPW attention section
            e_t = torch.bmm(h_t.unsqueeze(1), memory.permute(0, 2, 1))
            a_t = torch.softmax(e_t.squeeze(1), dim=1)  # softmax attention
            r_t = torch.bmm(a_t.unsqueeze(1), memory).squeeze(1)
            q_t_next = torch.cat((h_t, r_t), dim=1)

            return q_t_next, cell_state

        # perform t_steps towards permutation invariant representation
        for i in range(self.t_steps):
            q_star, cell_state = step(q_star, cell_state, memory)

        return q_star

    def init_hidden(self, embedded_inputs):
        """
        Initiate hidden units
        :param Tensor embedded_inputs: The embedded input of Pointer-Net
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = embedded_inputs.size(0)

        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).repeat(batch_size, self.hidden_dim * 2)
        c0 = self.h0.unsqueeze(0).repeat(batch_size, self.hidden_dim)

        return h0, c0


###############################################################################
###############################################################################
# Deep Sets (2017)
# Formalization of a simple way to obtain a permutation invariant representation
# of the input set.
# Original paper: https://arxiv.org/abs/1703.06114
# Implementation follows code provided by the author:
# https://github.com/manzilzaheer/DeepSets/blob/master/DigitSum/image_sum.ipynb
###############################################################################
###############################################################################


class DeepSets(nn.Module):
    """
    Following the paper author's implementation:
    https://github.com/manzilzaheer/DeepSets/blob/master/DigitSum/image_sum.ipynb
    It requires a stack of matrix multiplications, followed by e.g. mean or sum
    """

    def __init__(self, in_dim, out_dim, hidden):
        super(DeepSets, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, X):
        element_embedding = self.encode(X)
        return None, element_embedding


class DeepSetsPointerNetwork(nn.Module):
    """
    DeepSets encoder + PtrNet. Set representation concatenated to each
    element before passing to PtrNet.
    """

    def __init__(self,
                 elem_dims,
                 embedding_dim,
                 hidden_dim,
                 lstm_layers,
                 dropout,
                 bidir=False,
                 masking=True,
                 output_length=None,
                 embedding_by_dict=False,
                 embedding_by_dict_size=None):
        """
        Initiate Pointer-Net
        :param int elem_dims: dimensions of each invdividual set element
        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(DeepSetsPointerNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.masking = masking
        self.output_length = output_length
        self.embedding_by_dict = embedding_by_dict
        self.embedding_by_dict_size = embedding_by_dict_size
        if embedding_by_dict:
            self.embedding = nn.Embedding(embedding_by_dict_size,
                                          embedding_dim)
        else:
            self.embedding = nn.Linear(elem_dims, embedding_dim)
        self.set_embedding = DeepSets(embedding_dim, embedding_dim,
                                      embedding_dim)
        self.encoder = PointerEncoder(embedding_dim,  # concat -> times 2
                                      hidden_dim,
                                      lstm_layers,
                                      dropout,
                                      bidir)
        self.decoder = PointerDecoder(embedding_dim, hidden_dim,
                                      masking=self.masking,
                                      output_length=self.output_length)
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim),
                                        requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        """
        PointerNet - Forward-pass
        :param Tensor inputs: Input sequence (batch x seq_len x elem_dim)
        :return: Pointers probabilities and indices
        """
        # inputs: (batch x seq_len x elem_dim)
        # for TSP in 2D, elem_dim = 2
        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        # decoder_input0: (batch,  embedding)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        # inputs: (batch * seq_len, elem_dim)
        reshaped_inputs = inputs.view(batch_size * input_length, -1)

        # embedded_inputs: (batch, seq_len, embedding)
        if self.embedding_by_dict:
            reshaped_inputs = reshaped_inputs.long()
        else:
            reshaped_inputs = reshaped_inputs.float()
        embedded_inputs = self.embedding(reshaped_inputs).view(batch_size,
                                                               input_length, -1)

        # embed the entire set, perm-invar
        embedded_set, embedded_inputs = self.set_embedding(embedded_inputs)

        # # juggle dimensions of the set representation to match the batch of elems
        # embedded_set = embedded_set.unsqueeze(1).expand(-1,
        #                                                 embedded_inputs.size()[
        #                                                     1], -1)
        #
        # # conatenate
        # embedded_inputs_and_set = torch.cat((embedded_inputs, embedded_set), 2)

        # encoder_hidden0: [(num_lstms, batch_size,  hidden),
        #                   (num_lstms, batch_size,  hidden]
        # where the length depends on number of lstms & bidir
        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs)

        # encoder_outputs: (batch_size, seq_len, hidden)
        # encoder_hidden: [(num_lstms, batch_size, hidden),
        #                  (num_lstms, batch_size, hidden]
        # where the length depends on number of lstms & bidir
        encoder_outputs, encoder_hidden = self.encoder(embedded_inputs,
                                                       encoder_hidden0)

        if self.bidir:
            # last layer's h and c only, concatenated
            decoder_hidden0 = (
                torch.cat(
                    (encoder_hidden[0][-2:][0], encoder_hidden[0][-2:][1]),
                    dim=-1),
                torch.cat(
                    (encoder_hidden[1][-2:][0], encoder_hidden[1][-2:][1]),
                    dim=-1))
        else:
            # decoder_hidden0: ((batch, hidden),
            #                   (batch, hidden))
            decoder_hidden0 = (encoder_hidden[0][-1],
                               encoder_hidden[1][-1])
        (outputs, pointers), decoder_hidden = self.decoder(
            embedded_inputs,
            decoder_input0,
            decoder_hidden0,
            encoder_outputs)

        return outputs, pointers


###############################################################################
###############################################################################
# Set Transformer (2019)
# One of the latest set-encoding methods, that ensures permutation
# invariance, handles sets of varying lenghts. It's novel contribution
# is that it is able to encode higher-order interactions between elements.
# Original paper: https://arxiv.org/abs/1810.00825
# Implementation follows code provided by the authors:
# with: https://github.com/juho-lee/set_transformer
###############################################################################
###############################################################################


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


###############################################################################
###############################################################################
# Model functions
###############################################################################
###############################################################################


def unpack_model_config(model_cfg, exp_cfg):
    """
    Function for unpacking a model config and updating the main experiment
    config dictionary.
    """
    # general
    exp_cfg['learning_rate'] = model_cfg['train_learning_rate']
    exp_cfg['num_epochs'] = model_cfg['train_num_epochs']

    # element dimensionality at input
    exp_cfg['elem_dims'] = model_cfg['elem_dim']

    # element embedding
    exp_cfg['elem_encoder_type'] = model_cfg['elem_embedding_encoder_type']
    exp_cfg['elem_embedding_by_dict'] = model_cfg['elem_embedding_by_dict']
    exp_cfg['elem_embedding_by_dict_size'] = model_cfg['elem_embedding_by_dict_size']
    exp_cfg['elem_embedding_n_layers'] = model_cfg['elem_embedding_n_layers']
    exp_cfg['elem_embedding_dim'] = model_cfg['elem_embedding_dim']
    exp_cfg['elem_embedding_add_positonal_encoding'] = model_cfg['elem_embedding_add_positonal_encoding']

    # set encoding
    exp_cfg['set_encoder_type'] = model_cfg['set_encoder_type']
    exp_cfg['set_encoder_rpw_t_steps'] = model_cfg['set_encoder_rpw_t_steps']
    exp_cfg['set_embed_rnn_layers'] = model_cfg['set_encoder_rnn_layers']
    exp_cfg['set_embed_rnn_bidir'] = model_cfg['set_encoder_rnn_bidir']
    exp_cfg['set_embed_rnn_dropout'] = model_cfg['set_encoder_rnn_dropout']
    exp_cfg['set_pooling_type'] = model_cfg['set_encoder_pooling_type']
    exp_cfg['set_embed_num_heads'] = model_cfg['set_encoder_num_heads']
    exp_cfg['set_embed_num_seeds'] = model_cfg['set_encoder_num_seeds']
    exp_cfg['set_embed_dim'] = model_cfg['set_encoder_dim']
    exp_cfg['set_embed_n_layers'] = model_cfg['set_encoder_n_layers']
    exp_cfg['set_encoder_custom_attention_type'] = model_cfg['set_encoder_custom_attention_type']
    exp_cfg['set_encoder_custom_attention_n_layers'] = model_cfg['set_encoder_custom_attention_n_layers']

    # context rnn (optional)
    exp_cfg['context_rnn_used'] = model_cfg['context_rnn_used']
    exp_cfg['context_rnn_dim'] = model_cfg['context_rnn_dim']
    exp_cfg['context_rnn_layers'] = model_cfg['context_rnn_layers']
    exp_cfg['context_rnn_bidir'] = model_cfg['context_rnn_bidir']
    exp_cfg['context_rnn_dropout'] = model_cfg['context_rnn_dropout']

    # permutation module
    exp_cfg['permute_module_type'] = model_cfg['permute_module_type']
    exp_cfg['permute_module_is_concat'] = model_cfg['permute_module_is_concat']
    exp_cfg['permute_module_hidden_dim'] = model_cfg['permute_module_hidden_dim']
    exp_cfg['permute_module_dropout'] = model_cfg['permute_module_dropout']
    exp_cfg['permute_module_bidirectional'] = model_cfg['permute_module_bidirectional']
    exp_cfg['permute_module_masking'] = model_cfg['permute_module_masking']
    exp_cfg['permute_module_label_dim'] = model_cfg['permute_module_label_dim']
    exp_cfg['permute_module_pair_dim'] = model_cfg['permute_module_pair_dim']
    exp_cfg['permute_module_lamb_rela'] = model_cfg['permute_module_lamb_rela']

    # language model params (don't have to be present in every config)
    if 'sentence_ordering' in model_cfg.keys():
        exp_cfg['sentence_ordering'] = model_cfg['sentence_ordering']
    elif 'catalog_ordering' in model_cfg.keys():
        exp_cfg['catalog_ordering'] = model_cfg['catalog_ordering']

    # other
    exp_cfg['model_type'] = 'e_{}_s_{}_p_{}_ape_{}_ct_{}'.format(
        exp_cfg['elem_encoder_type'],
        exp_cfg['set_encoder_type'],
        exp_cfg['permute_module_type'],
        exp_cfg['elem_embedding_add_positonal_encoding'],
        exp_cfg['context_rnn_used']
    )

    return exp_cfg


def validate_model_config(config):
    """
    Take a config for a tsp experiment and validate it.
    Raise instructive errors if invalid.
    Bare-bones right now, maybe forever.
    """
    try:
        elem_encoder = config['elem_encoder_type']
    except KeyError:
        error_msg = 'No elem_encoder_type specified in the config!'
        logging.exception(error_msg)
        raise KeyError(error_msg)

    valid_elem_encoders = ['linear', 'settransformer']
    # valid_set_encoders = ['deepsets', 'settransformer', 'PROPOSED']
    # valid_permutation_modules = ['ptrnet', 'rpw', 'futurehistory', 'PROPOSED']

    if elem_encoder not in valid_elem_encoders:
        error_msg = "Unknown element encoder type! " \
                    "Check config['elem_encoder_type']" \
                    "field. Known types: {}".format(valid_elem_encoders)
        logging.exception(error_msg)
        raise Exception(error_msg)


def get_model(config):
    """
    Get a model specified by the configuration
    :param config: a dictionary with all model params
    :return: a model instance
    """
    # check if valid
    validate_model_config(config)

    # instantiate the model
    model = SetToSequence(
        elem_dim=config['elem_dims'],
        elem_embedding_encoder_type=config['elem_encoder_type'],
        elem_embedding_dim=config['elem_embedding_dim'],
        elem_embedding_by_dict=config['elem_embedding_by_dict'],
        elem_embedding_by_dict_size=config['elem_embedding_by_dict_size'],
        elem_embedding_n_layers=config['elem_embedding_n_layers'],
        elem_embedding_add_positonal_encoding=config['elem_embedding_add_positonal_encoding'],
        set_encoder_type=config['set_encoder_type'],
        set_encoder_custom_attention_type=config['set_encoder_custom_attention_type'],
        set_encoder_custom_attention_n_layers=config['set_encoder_custom_attention_n_layers'],
        set_encoder_rpw_t_steps=config['set_encoder_rpw_t_steps'],
        set_encoder_rnn_layers=config['set_embed_rnn_layers'],
        set_encoder_rnn_bidir=config['set_embed_rnn_bidir'],
        set_encoder_rnn_dropout=config['set_embed_rnn_dropout'],
        set_encoder_pooling_type=config['set_pooling_type'],
        set_encoder_num_heads=config['set_embed_num_heads'],
        set_encoder_num_seeds=config['set_embed_num_seeds'],
        set_encoder_dim=config['set_embed_dim'],
        set_encoder_n_layers=config['set_embed_n_layers'],
        context_rnn_used=config['context_rnn_used'],
        context_rnn_dim=config['context_rnn_dim'],
        context_rnn_layers=config['context_rnn_layers'],
        context_rnn_bidir=config['context_rnn_bidir'],
        context_rnn_dropout=config['context_rnn_dropout'],
        permute_module_type=config['permute_module_type'],
        permute_module_is_concat=config['permute_module_is_concat'],
        permute_module_hidden_dim=config['permute_module_hidden_dim'],
        permute_module_dropout=config['permute_module_dropout'],
        permute_module_bidir=config['permute_module_bidirectional'],
        permute_module_masking=config['permute_module_masking'],
        permute_module_label_dim=config['permute_module_label_dim'],
        permute_module_pair_dim=config['permute_module_pair_dim'],
        permute_module_lamb_rela=config['permute_module_lamb_rela'],
        sentence_ordering=config['sentence_ordering'] if 'sentence_ordering' in config.keys() else False,
        catalog_ordering=config['catalog_ordering'] if 'catalog_ordering' in config.keys() else False,
    )

    # float
    model.float()

    return model


# def train_epochs_old(a_model, a_model_path, a_model_optimizer, a_loss,
#                  a_train_dataloader,
#                  a_cv_dataloader, a_config, num_epochs,
#                  x_name, y_name, logger, tqdm_file, run_tests_function,
#                  allow_gpu=True,
#                  report_every_n_batches=1, validate_every_n_epochs=1,
#                  save_model_every_n_epochs=100):
#     """
#     Take a model, an optimizer and a num_epochs and train it.
#     """
#     # handle gpu/cpu
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     # cuda
#     if torch.cuda.is_available() and allow_gpu:
#         a_model.cuda()
#         net = torch.nn.DataParallel(a_model,
#                                     device_ids=range(
#                                         torch.cuda.device_count()))
#         cudnn.benchmark = True
#         logger.info('CUDA available, using GPU:')
#         logger.info(torch.cuda.get_device_name(0))
#     time.sleep(1)
#
#     # epochs
#     losses = []
#
#     for epoch in range(num_epochs):
#         epoch_losses = []
#         iterator = tqdm(a_train_dataloader, unit=' batches', file=tqdm_file)
#         logger.info('Epoch {}'.format(epoch + 1))
#
#         for i_batch, sample_batched in enumerate(iterator):
#             batch_losses = []
#             iterator.set_description(
#                 'Epoch {} / {}'.format(epoch + 1, num_epochs))
#
#             train_batch = Variable(sample_batched[x_name])
#             target_batch = Variable(sample_batched[y_name])
#
#             if torch.cuda.is_available() and allow_gpu:
#                 train_batch = train_batch.cuda()
#                 target_batch = target_batch.cuda()
#
#             o, p = a_model(train_batch)
#             o = o.contiguous().view(-1, o.size()[-1])
#
#             target_batch = target_batch.view(-1)
#
#             loss = a_loss(o, target_batch)
#             tracked_loss = loss.clone().detach().cpu().numpy()
#             losses.append(tracked_loss)
#             epoch_losses.append(tracked_loss)
#             batch_losses.append(tracked_loss)
#
#             a_model_optimizer.zero_grad()
#             loss.backward()
#             a_model_optimizer.step()
#
#             # report avg loss for batch
#             batch_loss = sum(batch_losses) / len(batch_losses)
#             iterator.set_postfix(avg_loss=' {:.5f}'.format(batch_loss))
#
#             if i_batch % report_every_n_batches == 0:
#                 logger.info(
#                     'Epoch {} Batch {}, average loss {:.5f}'.format(epoch + 1,
#                                                                     i_batch + 1,
#                                                                     batch_loss))
#         epoch_loss = sum(epoch_losses) / len(epoch_losses)
#         logger.info('Epoch {} average loss {:.5f}'.format(epoch + 1,
#                                                           epoch_loss))
#
#         # report cv performance
#         if (epoch + 1) % validate_every_n_epochs == 0:
#             a_model.eval()
#             logger.info('Validation at epoch {} starting ...'.format(epoch + 1))
#             _ = run_tests_function(a_model, a_cv_dataloader, logger, a_config,
#                                    batched=True)
#             logger.info('Validation at epoch {} finished.'.format(epoch + 1))
#             a_model.train()
#
#         # save model
#         if (epoch + 1) % save_model_every_n_epochs == 0:
#             logger.info('Model saving ... ')
#             a_model.to('cpu')
#             torch.save(a_model,
#                        a_model_path + '_epoch_{}.pth'.format(epoch + 1))
#             a_model.to(device)
#
#     # report final loss
#     logger.info('Final training loss: {}'.format(loss))
#
#     # return last loss
#     return loss, losses


def train_epochs(a_model, a_model_path, a_model_optimizer, a_loss,
                 a_train_dataloader,
                 a_cv_dataloader, a_config, num_epochs,
                 x_name, y_name, logger, tqdm_file, run_tests_function,
                 allow_gpu=True, is_futurehistory=False, is_sentence_ordering=False,
                 report_every_n_batches=1, validate_every_n_epochs=1,
                 save_model_every_n_epochs=100):
    """
    Take a model, an optimizer and a num_epochs and train it.
    """
    # handle gpu/cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # cuda
    if torch.cuda.is_available() and allow_gpu:
        a_model.cuda()
        net = torch.nn.DataParallel(a_model,
                                    device_ids=range(
                                        torch.cuda.device_count()))
        cudnn.benchmark = True
        logger.info('CUDA available, using GPU:')
        logger.info(torch.cuda.get_device_name(0))
    time.sleep(1)

    # F+H handles loss internally
    if is_futurehistory:
        a_model.decoder.equip(a_loss)

    # lose any gradients
    a_model.zero_grad()

    # epochs
    losses = []

    for epoch in range(num_epochs):
        epoch_losses = []
        iterator = tqdm(a_train_dataloader, unit=' batches', file=tqdm_file)
        logger.info('Epoch {}'.format(epoch + 1))

        for i_batch, sample_batched in enumerate(iterator):
            batch_losses = []
            iterator.set_description(
                'Epoch {} / {}'.format(epoch + 1, num_epochs))

            # handle sentence ordering format
            if is_sentence_ordering:
                # move batch to gpu, if available
                if allow_gpu and device == 'cuda':
                    sample_batched = {k: v.to(device) for k, v in sample_batched.items()}

                # handle huggingface batches
                train_batch = sample_batched
                target_batch = sample_batched['label']

            # if not sentence ordering, proceed as normally
            else:
                train_batch = Variable(sample_batched[x_name])
                target_batch = Variable(sample_batched[y_name])

                if torch.cuda.is_available() and allow_gpu:
                    train_batch = train_batch.cuda()
                    target_batch = target_batch.cuda()

            if is_futurehistory:
                loss, (point_loss, rela_loss, left1_loss, left2_loss) = a_model(
                    train_batch, target_batch)
            else:
                o, p = a_model(train_batch)
                o = o.contiguous().view(-1, o.size()[-1])
                target_batch = target_batch.view(-1)
                loss = a_loss(o, target_batch)

            tracked_loss = loss.clone().detach().cpu().numpy()
            losses.append(tracked_loss)
            epoch_losses.append(tracked_loss)
            batch_losses.append(tracked_loss)

            a_model_optimizer.zero_grad()
            loss.backward()
            a_model_optimizer.step()

            # report avg loss for batch
            batch_loss = sum(batch_losses) / len(batch_losses)
            iterator.set_postfix(avg_loss=' {:.5f}'.format(batch_loss))

            if i_batch % report_every_n_batches == 0:
                logger.info(
                    'Epoch {} Batch {}, average loss {:.5f}'.format(epoch + 1,
                                                                    i_batch + 1,
                                                                    batch_loss))
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info('Epoch {} average loss {:.5f}'.format(epoch + 1,
                                                          epoch_loss))

        # report cv performance, unless final epoch
        if (epoch + 1) % validate_every_n_epochs == 0 and (epoch + 1) != num_epochs:
            a_model.eval()
            logger.info('Validation at epoch {} starting ...'.format(epoch + 1))
            _ = run_tests_function(a_model, a_cv_dataloader, logger, a_config,
                                   batched=True)
            logger.info('Validation at epoch {} finished.'.format(epoch + 1))
            a_model.train()

        # save model
        if (epoch + 1) % save_model_every_n_epochs == 0:
            logger.info('Model saving ... ')
            a_model.to('cpu')
            torch.save(a_model,
                       a_model_path + '_epoch_{}.pth'.format(epoch + 1))
            a_model.to(device)

    # report final loss
    logger.info('Final training loss: {}'.format(loss))

    # return last loss
    return loss, losses
