# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from Transformer import TransformerEncoder, TransformerEncoderLayer
from nn_utils import PositionalEncoding, has_nan
from dataset_utils import len_to_mask


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
    ):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class BERTEmbeddings(nn.Module):
    def __init__(self, config):
        super(BERTEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERTSelfAttention(nn.Module):
    def __init__(self, config):
        super(BERTSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BERTSelfOutput(nn.Module):
    def __init__(self, config):
        super(BERTSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTAttention(nn.Module):
    def __init__(self, config):
        super(BERTAttention, self).__init__()
        self.self = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BERTIntermediate(nn.Module):
    def __init__(self, config):
        super(BERTIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):
    def __init__(self, config):
        super(BERTOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTLayer(nn.Module):
    def __init__(self, config):
        super(BERTLayer, self).__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BERTEncoder(nn.Module):
    def __init__(self, config):
        super(BERTEncoder, self).__init__()
        layer = BERTLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BERTPooler(nn.Module):
    def __init__(self, config):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config: BertConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(BertModel, self).__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        all_encoder_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return all_encoder_layers, pooled_output


class BertForSequenceClassificationEntityAvg(nn.Module):
    """
     Not Implementable for DialogRE since some entiites not appear in context.
    """
    def __init__(self, config, args=None, num_labels=1, entity_drop=0.1):
        super(BertForSequenceClassificationEntityAvg, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_entity = nn.Dropout(entity_drop)
        self.classifier = nn.Linear(config.hidden_size * 3, num_labels * 36)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        labels=None,
        n_class=1,
        b_mask=None,
        c_mask=None,
    ):
        seq_length = input_ids.size(2)
        all_encoder_layers, pooled_output = self.bert(
            input_ids.view(-1, seq_length),
            token_type_ids.view(-1, seq_length),
            attention_mask.view(-1, seq_length),
        )
        enc_mem = all_encoder_layers[-1]  # [bsz, seq_len, hid_size]

        # print('enc_mem', enc_mem.size())
        # print('b_mask', b_mask.size())              # [bsz, 1, seq_len]
        # print('c_mask', c_mask.size())              # [bsz, 1, seq_len]
        # print('pooled_out', pooled_output.size())   # [bsz, hid_size]
        b_num = b_mask.sum(-1)  # [bsz, 1]
        b_mask = b_mask.squeeze(1).unsqueeze(-1).repeat(1, 1, enc_mem.size(-1))
        c_num = c_mask.sum(-1)  # [bsz, 1]
        c_mask = c_mask.squeeze(1).unsqueeze(-1).repeat(1, 1, enc_mem.size(-1))
        
        # print('b_mask', b_mask.size())
        # print('c_mask', c_mask.size())
        # print('b_num', b_num.size(), b_num)
        # print('c_num', c_num.size(), c_num)
        b_pooled = self.dropout_entity((enc_mem * b_mask).sum(dim=1) / b_num)  # [bsz, hid_size]
        c_pooled = self.dropout_entity((enc_mem * c_mask).sum(dim=1) / c_num)
        # print('b_pooled', b_pooled.size())
        # print('c_pooled', c_pooled.size())
        # exit()

        pooled_output = self.dropout(pooled_output)
        pooled_output_catted = torch.cat(
            [pooled_output, b_pooled, c_pooled], dim=-1
        )  # [bsz, hid_size * 3]
        logits = self.classifier(pooled_output_catted)
        logits = logits.view(-1, 36)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.view(-1, 36)
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class BertForSequenceClassificationEntityMax(nn.Module):
    def __init__(self, config, args=None, num_labels=1, entity_drop=0.1):
        super(BertForSequenceClassificationEntityMax, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_entity = nn.Dropout(entity_drop)
        self.classifier = nn.Linear(config.hidden_size * 3, num_labels * 36)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        labels=None,
        n_class=1,
        b_mask=None,
        c_mask=None,
    ):
        seq_length = input_ids.size(2)
        all_encoder_layers, pooled_output = self.bert(
            input_ids.view(-1, seq_length),
            token_type_ids.view(-1, seq_length),
            attention_mask.view(-1, seq_length),
        )
        enc_mem = all_encoder_layers[-1]              # [bsz, seq_len, hid_size]

        # print('enc_mem', enc_mem.size())
        # print('b_mask', b_mask.size())              # [bsz, 1, seq_len]
        # print('c_mask', c_mask.size())              # [bsz, 1, seq_len]
        # print('pooled_out', pooled_output.size())   # [bsz, hid_size]

        b_mask = b_mask.squeeze(1).unsqueeze(-1).repeat(1, 1, enc_mem.size(-1))
        c_mask = c_mask.squeeze(1).unsqueeze(-1).repeat(1, 1, enc_mem.size(-1))
        b_pooled = self.dropout_entity(torch.max(enc_mem * b_mask, dim=1).values)  # [bsz, hid_size]
        c_pooled = self.dropout_entity(torch.max(enc_mem * c_mask, dim=1).values)
        # print('b_pooled', b_pooled.size())
        # print('c_pooled', c_pooled.size())
        # exit()

        pooled_output = self.dropout(pooled_output)
        pooled_output_catted = torch.cat(
            [pooled_output, b_pooled, c_pooled], dim=-1
        )  # [bsz, hid_size * 3]
        logits = self.classifier(pooled_output_catted)
        logits = logits.view(-1, 36)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.view(-1, 36)
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class BertForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels * 36)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        labels=None,
        n_class=1,
        b_mask=None,
        c_mask=None,
    ):
        seq_length = input_ids.size(2)
        _, pooled_output = self.bert(
            input_ids.view(-1, seq_length),
            token_type_ids.view(-1, seq_length),
            attention_mask.view(-1, seq_length),
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = logits.view(-1, 36)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.view(-1, 36)
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class HierAMRBertForSequenceClassificationEntityMax(nn.Module):
    def __init__(self, config, args=None, num_labels=1, relation_vocab=None):
        super(HierAMRBertForSequenceClassificationEntityMax, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_entity = nn.Dropout(args.entity_drop)
        self.classifier = nn.Linear(config.hidden_size + args.d_concept * 2, num_labels * 36)
        assert relation_vocab is not None
        self.GraphEnc = GraphTransformer(args, relation_vocab)
        if config.hidden_size != args.d_concept:
            self.enc_map = nn.Linear(config.hidden_size, args.d_concept)
        else:
            self.enc_map = None
        self.EncoderNorm = nn.LayerNorm(args.d_concept)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)

    def forward(
        self,
        input_ids,
        input_len,
        token_type_ids,
        attention_mask,
        labels=None,
        n_class=1,
        b_mask=None,
        c_mask=None,
        rel_mask=None,
        word_rel=None,
        con_ids=None,
        con_len=None,
        con_path=None,
        b_mask_con=None,
        c_mask_con=None
    ):
        seq_length = input_ids.size(2)
        all_encoder_layers, pooled_output = self.bert(
            input_ids.view(-1, seq_length),
            token_type_ids.view(-1, seq_length),
            attention_mask.view(-1, seq_length),
        )
        enc_mem = all_encoder_layers[-1]  # [bsz, seq_len, hid_size]
        enc_mem = enc_mem[:, :491, :]
        if self.enc_map is not None:
            enc_mem = self.enc_map(enc_mem)
        # print('enc_mem', enc_mem.size())
        # if self.GraphEnc is not None:
        enc_mem_new = self.GraphEnc(enc_mem, input_len, word_rel)
        # print('enc_mem_new', enc_mem.size())
        # print('b_mask', b_mask.size())              # [bsz, 1, seq_len]
        # print('c_mask', c_mask.size())              # [bsz, 1, seq_len]
        # print('pooled_out', pooled_output.size())   # [bsz, hid_size]
        enc_mem = self.EncoderNorm(enc_mem + enc_mem_new)

        b_mask = b_mask.squeeze(1).unsqueeze(-1).repeat(1, 1, enc_mem.size(-1))[:, :491, :]
        c_mask = c_mask.squeeze(1).unsqueeze(-1).repeat(1, 1, enc_mem.size(-1))[:, :491, :]
        b_pooled = self.dropout_entity(torch.max(enc_mem * b_mask, dim=1).values)  # [bsz, hid_size]
        c_pooled = self.dropout_entity(torch.max(enc_mem * c_mask, dim=1).values)
        # print('b_pooled', b_pooled.size())
        # print('c_pooled', c_pooled.size())
        # exit()

        pooled_output = self.dropout(pooled_output)
        pooled_output_catted = torch.cat(
            [pooled_output, b_pooled, c_pooled], dim=-1
        )  # [bsz, hid_size * 3]
        logits = self.classifier(pooled_output_catted)
        logits = logits.view(-1, 36)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.view(-1, 36)
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class DualAMRBertForSequenceClassificationEntityMax(nn.Module):
    def __init__(
        self,
        config,
        args=None,
        num_labels=1,
        relation_vocab=None,
        concept_vocab=None,
        path_vocab=None,
    ):
        super(DualAMRBertForSequenceClassificationEntityMax, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_entity = nn.Dropout(args.entity_drop)
        self.classifier = nn.Linear(config.hidden_size + args.d_concept * 2, num_labels * 36)
        assert concept_vocab is not None
        assert path_vocab is not None
        self.GraphEnc = GraphTransformer(args, path_vocab)
        self.concept_emb = nn.Embedding(len(concept_vocab), args.d_concept)
        if config.hidden_size != args.d_concept:
            self.enc_map = nn.Linear(config.hidden_size, args.d_concept)
        else:
            self.enc_map = None
        self.EncoderNorm = nn.LayerNorm(args.d_concept)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)

    def forward(
        self,
        input_ids,
        input_len,
        token_type_ids,
        attention_mask,
        labels=None,
        n_class=1,
        b_mask=None,
        c_mask=None,
        rel_mask=None,
        word_rel=None,
        con_ids=None,
        con_len=None,
        con_path=None,
        b_mask_con=None,
        c_mask_con=None
    ):
        seq_length = input_ids.size(2)
        all_encoder_layers, pooled_output = self.bert(
            input_ids.view(-1, seq_length),
            token_type_ids.view(-1, seq_length),
            attention_mask.view(-1, seq_length),
        )
        enc_mem = all_encoder_layers[-1]                        # [bsz, seq_len, hid_size]
        b_mask = b_mask.squeeze(1).unsqueeze(-1).repeat(1, 1, enc_mem.size(-1))
        c_mask = c_mask.squeeze(1).unsqueeze(-1).repeat(1, 1, enc_mem.size(-1))
        b_pooled = self.dropout_entity(torch.max(enc_mem * b_mask, dim=1).values)  # [bsz, hid_size]
        c_pooled = self.dropout_entity(torch.max(enc_mem * c_mask, dim=1).values)
        if self.enc_map is not None:
            b_pooled = self.enc_map(b_pooled)
            c_pooled = self.enc_map(c_pooled)
        # print('con_ids', con_ids.size())
        # print('con_len', con_len.size())
        # print('con_path', con_path.size())
        # print('con_b_mask', b_mask_con.size())
        # print('con_c_mask', c_mask_con.size())
        # exit()

        con_emb = self.concept_emb(con_ids.squeeze(1))
        graph_mem = self.GraphEnc(con_emb, con_len.squeeze(1), con_path.squeeze(1))     # [bsz, graph_size, d_concept]
        b_mask_con = b_mask_con.squeeze(1).unsqueeze(-1).repeat(1, 1, graph_mem.size(-1))
        c_mask_con = c_mask_con.squeeze(1).unsqueeze(-1).repeat(1, 1, graph_mem.size(-1))
        b_pooled_con = self.dropout_entity(torch.max(graph_mem * b_mask_con, dim=1).values)
        c_pooled_con = self.dropout_entity(torch.max(graph_mem * c_mask_con, dim=1).values)

        b_pooled = self.EncoderNorm(b_pooled + b_pooled_con)
        c_pooled = self.EncoderNorm(c_pooled + c_pooled_con)

        pooled_output = self.dropout(pooled_output)
        pooled_output_catted = torch.cat(
            [pooled_output, b_pooled, c_pooled], dim=-1
        )  # [bsz, hid_size * 3]
        logits = self.classifier(pooled_output_catted)
        logits = logits.view(-1, 36)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.view(-1, 36)
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class GraphTransformer(nn.Module):
    def __init__(self, config, relation_vocab):
        super(GraphTransformer, self).__init__()

        self.config = config
        self.relation_vocab = relation_vocab
        self.n_layer = getattr(config, "g_num_layer", 2)
        self.use_pe = getattr(config, "g_pe", True)
        self.d_relation = getattr(config, "d_relation", 64)
        self.d_concept = getattr(config, "d_concept", 768)
        self.g_n_head = self.d_concept // self.d_relation
        self.dropout = getattr(config, "dropout", 0.1)
        assert relation_vocab is not None
        self.relation_embedding = nn.Embedding(len(self.relation_vocab), self.d_relation)
        assert self.d_relation * self.g_n_head == self.d_concept

        self.position_encoder = PositionalEncoding(self.d_concept)
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_concept,
            heads=self.g_n_head,
            d_ff=getattr(config, "g_d_ff", 1024),
            dropout=self.dropout,
            att_drop=self.dropout,
            use_structure=True,
        )
        encoder_norm = nn.LayerNorm(self.d_concept)
        self.encoder = TransformerEncoder(encoder_layer, self.n_layer, encoder_norm)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_emb, src_length, relations):
        # src_emb:  [batch, seq, dim]
        # print("input_size", src_emb.size())
        bsz, max_seq_len, hid_size = src_emb.size()
        # print('max_seq_len:', max_seq_len)
        src_mask = torch.from_numpy(len_to_mask(src_length, max_seq_len)).to(src_emb.device)
        src_mask_inv = src_mask == 0
        relations = relations.squeeze(1)
        structure_emb = self.relation_embedding(relations)  # [batch, seq, seq, s_dim]
        # print('structure size:', structure_emb.size())
        if self.use_pe:
            src_emb = self.position_encoder(src_emb.transpose(0, 1)).transpose(
                0, 1
            )  # [batch, seq, dim]
        src_emb = self.encoder(
            src_emb, src_key_padding_mask=src_mask_inv, structure=structure_emb
        )  # [batch, seq, dim]
        assert has_nan(src_emb) is False
        return src_emb
