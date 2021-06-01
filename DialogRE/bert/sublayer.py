# coding:utf-8
import math
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):
    """
  Multi-Head Attention module from
  "Attention is All You Need"
  :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

  Similar to standard `dot` attention but uses
  multiple attention distributions simulataneously
  to select relevant items.

  Args:
     head_count (int): number of parallel heads
     model_dim (int): the dimension of keys/values/queries,
         must be divisible by head_count
     dropout (float): dropout parameter
  """

    def __init__(
        self,
        head_count,
        kv_dim,
        query_dim=512,
        dropout=0.1,
        use_structure=False,
        bias=True,
        alpha=1.0,
        beta=1.0,
    ):
        assert kv_dim % head_count == 0
        self.dim_per_head = kv_dim // head_count
        self.kv_dim = kv_dim
        self.query_dim = query_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(kv_dim, head_count * self.dim_per_head, bias=bias)
        self.linear_values = nn.Linear(kv_dim, head_count * self.dim_per_head, bias=bias)
        self.linear_query = nn.Linear(query_dim, head_count * self.dim_per_head, bias=bias)

        if use_structure:
            self.linear_structure_k = nn.Linear(self.dim_per_head, self.dim_per_head)
            self.linear_structure_v = nn.Linear(self.dim_per_head, self.dim_per_head)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(kv_dim, kv_dim)
        # self.final_linear = nn.Linear(kv_dim, query_dim)
        self.alpha = alpha
        self.beta = beta
        self.use_structure = use_structure
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.linear_query.weight)
        xavier_uniform_(self.linear_keys.weight)
        xavier_uniform_(self.linear_values.weight)

    def forward(
        self,
        query,
        key,
        value,
        structure=None,
        mask=None,
        key_padding_mask=None,
        layer_cache=None,
        type=None,
    ):
        """
        Compute the context vector and the attention vectors.

        Args:
        key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
        value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
        query (`FloatTensor`): set of `query_len`
                query vectors  `[batch, query_len, dim]`
        structure (`FloatTensor`): set of `query_len`
                query vectors  `[batch, query_len, query_len, dim]`

        mask: binary key2key mask indicating which keys have
                non-zero attention `[batch, key_len, key_len]`
        key_padding_mask: binary padding mask indicating which keys have
                non-zero attention `[batch, key_len]`
        
        Returns:
        (`FloatTensor`, `FloatTensor`) :
        * output context vectors `[batch, query_len, dim]`
        * one of the attention vectors `[batch, query_len, key_len]`
        """

        # CHECKS
        """
        batch, k_len, d = key.size()
        batch_, k_len_, d_ = value.size()
        aeq(batch, batch_)
        aeq(k_len, k_len_)
        aeq(d, d_)
        batch_, q_len, d_ = query.size()
        aeq(batch, batch_)
        aeq(d, d_)
        aeq(self.kv_dim % 8, 0)
        if mask is not None:
        batch_, q_len_, k_len_ = mask.size()
        aeq(batch_, batch)
        aeq(k_len_, k_len)
        aeq(q_len_ == q_len)
        print('q_len_mask: {}, q_len:{}'.format(q_len_, q_len))
        """
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)
        # print('key_size', key.size())
        # print('value_size', value.size())

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:  # for decoder self-attn
            if type == "self":
                query, key, value = (  # [bsz, seq_len, H]
                    self.linear_query(query),
                    self.linear_keys(query),
                    self.linear_values(query),
                )
                if structure is not None and self.use_structure:
                    structure_k, structure_v = (
                        self.linear_structure_k(structure),  # [bsz, seq_len, seq_len, H]
                        self.linear_structure_v(structure),  # []
                    )
                else:
                    structure_k = None
                    structure_v = None

                key = shape(key)  # [bsz, nhead, key_len, H_head]
                value = shape(value)  # [bsz, nhead, value_len, H_head]

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat((layer_cache["self_keys"].to(device), key), dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat((layer_cache["self_values"].to(device), value), dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value

            elif type == "context":  # for decoder context-attn
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key), self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"], layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key), self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:  # encoder/decoder self/context attn
            key = self.linear_keys(key)
            value = self.linear_values(value)
            # print('input:', query.size())
            # print('Linear:', self.linear_query)
            query = self.linear_query(query)
            if structure is not None and self.use_structure:
                structure_k, structure_v = (
                    self.linear_structure_k(structure),
                    self.linear_structure_v(structure),
                )
            else:
                structure_k = None
                structure_v = None

            key = shape(key)  # [batch_size, nhead, key_len, dim]
            value = shape(value)

        query = shape(query)  # [batch_size, nhead, key_len, dim]
        # print('key, query', key.size(), query.size())

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)  # attention scale
        scores = torch.matmul(query, key.transpose(2, 3))  # [batch_size, nhead, query_len, key_len]
        # print('scores', scores.size())

        if structure_k is not None:  # [batch_size, seq_len, seq_len, dim]
            q = query.transpose(1, 2)  # [batch_size, seq_len, nhead, dim]
            # print(q.size(), structure_k.transpose(2,3).size())
            scores_k = torch.matmul(
                q, structure_k.transpose(2, 3)
            )  # [batch_size, seq_len, nhead, seq_len]
            scores_k = scores_k.transpose(1, 2)  # [batch_size, nhead, seq_len, seq_len]
            # print (scores.size(),scores_k.size())
            scores = scores + self.alpha * scores_k

        if key_padding_mask is not None:  # padding mask
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # `[B, 1, 1, seq_len]`
            # print('key_padding_mask', key_padding_mask.size())
            scores = scores.masked_fill(key_padding_mask.bool(), -1e4)  # -1e4 allows fp16
            # print('scores_masked', scores)

        if mask is not None:  # key-to-key mask
            mask = mask.unsqueeze(1)  # `[B, 1, seq_len, seq_len]`
            scores = scores.masked_fill(mask.bool(), -1e4)  # -1e4 allows fp16

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        # print(drop_attn[0][0][3])
        context = torch.matmul(drop_attn, value)

        if structure_v is not None:
            drop_attn_v = drop_attn.transpose(1, 2)  # [batch_size, seq_len, nhead, seq_len]
            context_v = torch.matmul(
                drop_attn_v, structure_v
            )  # [batch_size, seq_len, seq_len, dim]
            context_v = context_v.transpose(1, 2)  # [batch_size, nhead, seq_len, dim]
            context = context + self.beta * context_v

        context = unshape(context)
        output = self.final_linear(context)

        # Return one attn
        first_head_attn = drop_attn.view(batch_size, head_count, query_len, key_len)[
            :, 0, :, :
        ].contiguous()

        return output, first_head_attn


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Layer definition.

        Args:
            input: [ batch_size, input_len, model_dim ]


        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


"""Global attention modules (Luong / Bahdanau)"""

# This class is mainly used by decoder.py for RNNs but also
# by the CNN / transformer decoder when copy attention is used
# CNN has its own attention mechanism ConvMultiStepAttention
# Transformer has its own MultiHeadedAttention


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class GlobalAttention(nn.Module):
    r"""
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = \sum_{j=1}^{\text{SeqLength}} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`\text{score}(H_j,q) = H_j^T q`
       * general: :math:`\text{score}(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`\text{score}(H_j, q) = v_a^T \text{tanh}(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]
       attn_func (str): attention function to use, options [softmax,sparsemax]

    """

    def __init__(self, dim, coverage=False, attn_type="dot", attn_func="softmax"):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        assert attn_type in [
            "dot",
            "general",
            "mlp",
        ], "Please select a valid attention type (got {:s}).".format(attn_type)
        self.attn_type = attn_type
        assert attn_func in ["softmax"], "Please select a valid attention function."
        self.attn_func = attn_func

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (FloatTensor): sequence of queries ``(batch, tgt_len, dim)``
          h_s (FloatTensor): sequence of sources ``(batch, src_len, dim``

        Returns:
          FloatTensor: raw attention scores (unnormalized) for each src index
            ``(batch, tgt_len, src_len)``
        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, memory_lengths=None, coverage=None):
        """

        Args:
          source (FloatTensor): query vectors ``(batch, tgt_len, dim)``
          memory_bank (FloatTensor): source vectors ``(batch, src_len, dim)``
          memory_lengths (LongTensor): the source context lengths ``(batch,)``
          coverage (FloatTensor): None (not supported yet)

        Returns:
          (FloatTensor, FloatTensor):

          * Computed vector ``(tgt_len, batch, dim)``
          * Attention distribtutions for each query
            ``(tgt_len, batch, src_len)``
        """

        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        if coverage is not None:
            batch_, source_l_ = coverage.size()

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank += self.linear_cover(cover).view_as(memory_bank)
            memory_bank = torch.tanh(memory_bank)

        # compute attention scores, as in Luong et al.
        align = self.score(source, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)                    # Make it broadcastable.
            align.masked_fill_(~mask, -float("inf"))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(align.view(batch * target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank)

        # concatenate
        concat_c = torch.cat([c, source], 2).view(batch * target_l, dim * 2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            batch_, source_l_ = align_vectors.size()
        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()
            # Check output sizes
            target_l_, batch_, dim_ = attn_h.size()
            target_l_, batch_, source_l_ = align_vectors.size()

        return attn_h, align_vectors


class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input_feed, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input_feed, (h_0[i], c_0[i]))
            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input_feed, (h_1, c_1)


class StackedGRU(nn.Module):
    """
    Our own implementation of stacked GRU.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input_feed, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input_feed, hidden[0][i])
            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)
        return input_feed, (h_1,)


class CopyGenerator(nn.Module):
    """An implementation of pointer-generator networks
    :cite:`DBLP:journals/corr/SeeLM17`.

    These networks consider copying words
    directly from the source sequence.

    The copy generator is an extended version of the standard
    generator that computes three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`


    .. mermaid::

       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O


    Args:
       input_size (int): size of input representation
       output_size (int): size of output vocabulary
       pad_idx (int)
    """

    def __init__(self, input_size, output_size, pad_idx):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear_copy = nn.Linear(input_size, 1)
        self.pad_idx = pad_idx

    def forward(self, hidden, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.

        Args:
           hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
           attn (FloatTensor): attn for each ``(batch x tlen, input_size)``
           src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the "extended" vocab containing.
               ``(src_len, batch, extra_words)``
        """

        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.pad_idx] = -float('inf')
        prob = torch.softmax(logits, 1)

        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(hidden))
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy)
        mul_attn = torch.mul(attn, p_copy)
        copy_prob = torch.bmm(
            mul_attn.view(-1, batch, slen).transpose(0, 1),
            src_map.transpose(0, 1)
        ).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return torch.cat([out_prob, copy_prob], 1)