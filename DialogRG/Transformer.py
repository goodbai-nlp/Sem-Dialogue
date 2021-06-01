# coding:utf-8
"""Base class for encoders and generic multi encoders."""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from nn_utils import _get_activation_fn, _get_clones
from sublayer import MultiHeadedAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        heads,
        d_ff=2048,
        dropout=0.1,
        att_drop=0.1,
        use_structure=True,
        alpha=1.0,
        beta=1.0,
        activation="relu",
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, query_dim=d_model, dropout=att_drop, use_structure=use_structure, alpha=alpha, beta=beta
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, inputs, mask=None, key_padding_mask=None, structure=None):
        """
            Args:
            input (`FloatTensor`): set of `key_len`
                    key vectors `[batch, seq_len, H]`
            mask: binary key2key mask indicating which keys have
                    non-zero attention `[batch, seq_len, seq_len]`
            key_padding_mask: binary padding mask indicating which keys have
                    non-zero attention `[batch, 1, seq_len]`
            return:
            res:  [batch, seq_len, H]
        """
        src = inputs
        src2, _ = self.self_attn(
            src, src, src, mask=mask, key_padding_mask=key_padding_mask, structure=structure,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff=2048, dropout=0.1, att_drop=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, query_dim=d_model, dropout=att_drop, use_structure=False)
        self.cross_attn = MultiHeadedAttention(
            heads, d_model, query_dim=d_model, dropout=att_drop, use_structure=False
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
            Args:
            tgt (`FloatTensor`): set of `key_len`
                    key vectors `[batch, tgt_len, H]`
            memory (`FloatTensor`): set of `key_len`
                    key vectors `[batch, src_len, H]`
            tgt_mask: binary key2key mask indicating which keys have
                    non-zero attention `[tgt_len, tgt_len]`
            memory_mask: binary key2key mask indicating which keys have
                    non-zero attention `[tgt_len, src_len]`
            tgt_key_padding_mask: binary padding mask indicating which keys have
                    non-zero attention `[batch, tgt_len]`
            memory_key_padding_mask: binary padding mask indicating which keys have
                    non-zero attention `[batch, src_len]`
            return:
            res:  [batch, tgt_len, H]
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(
            tgt, memory, memory, mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class DoubleAttnTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, d_sent, d_con=512, heads=8, d_ff=2048, dropout=0.1, att_drop=0.1, activation="relu", dual_enc=True):
        super(DoubleAttnTransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, query_dim=d_model, dropout=att_drop, use_structure=False)
        self.dual_enc = dual_enc
        self.d_sent = d_sent
        self.d_con = d_con
        self.sent_cross_attn = MultiHeadedAttention(
            heads, d_sent, query_dim=d_model, dropout=att_drop, use_structure=False
        )
        if self.d_sent != d_model and not dual_enc:
            self.kv_map = nn.Linear(self.d_sent, d_model)
        else:
            self.kv_map = None
        n_graph_head = 4 if self.d_con != 512 else 8
        if dual_enc:
            self.graph_cross_attn = MultiHeadedAttention(
                n_graph_head, self.d_con, query_dim=d_model, dropout=att_drop, use_structure=False
            )
            self.fuse_linear = nn.Linear(self.d_sent + self.d_con, d_model)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        tgt: Tensor,
        sent_memory: Tensor,
        graph_memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        sent_memory_mask: Optional[Tensor] = None,
        graph_memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        sent_memory_key_padding_mask: Optional[Tensor] = None,
        graph_memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
            Args:
            tgt (`FloatTensor`): set of `key_len`
                    key vectors `[batch, tgt_len, H]`
            memory (`FloatTensor`): set of `key_len`
                    key vectors `[batch, src_len, H]`
            tgt_mask: binary key2key mask indicating which keys have
                    non-zero attention `[tgt_len, tgt_len]`
            sent_memory_mask: binary key2key mask indicating which keys have
                    non-zero attention `[tgt_len, src_sent_len]`
            graph_memory_mask: binary key2key mask indicating which keys have
                    non-zero attention `[tgt_len, src_graph_len]`
            tgt_key_padding_mask: binary padding mask indicating which keys have
                    non-zero attention `[batch, tgt_len]`
            sent_memory_key_padding_mask: binary padding mask indicating which keys have
                    non-zero attention `[batch, src_sent_len]`
            graph_memory_key_padding_mask: binary padding mask indicating which keys have
                    non-zero attention `[batch, src_graph_len]`
            return:
            res:  [batch, tgt_len, H]
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        sent_tgt2 = self.sent_cross_attn(
            tgt, sent_memory, sent_memory, mask=sent_memory_mask, key_padding_mask=sent_memory_key_padding_mask)[0]
        if self.dual_enc:
            graph_tgt2 = self.graph_cross_attn(
                tgt, graph_memory, graph_memory, mask=graph_memory_mask, key_padding_mask=graph_memory_key_padding_mask)[0]
            tgt2 = self.fuse_linear(torch.cat([sent_tgt2, graph_tgt2], dim=-1))
        else:
            if self.kv_map is not None:
                sent_tgt2 = self.kv_map(sent_tgt2)
            tgt2 = sent_tgt2
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class DoubleAttnTransformerDecoderLayerSentFirst(nn.Module):
    def __init__(self, d_model, d_enc, heads, d_ff=2048, dropout=0.1, att_drop=0.1, activation="relu", dual_enc=True):
        super(DoubleAttnTransformerDecoderLayerSentFirst, self).__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, query_dim=d_model, dropout=att_drop, use_structure=False)
        self.dual_enc = dual_enc
        self.d_enc = d_enc

        self.sent_cross_attn = MultiHeadedAttention(
            heads, d_model, query_dim=d_model, dropout=att_drop, use_structure=False
        )
        if dual_enc:
            self.graph_cross_attn = MultiHeadedAttention(
                heads, self.d_enc - d_model, query_dim=d_model, dropout=att_drop, use_structure=False
            )
            # self.fuse_linear = nn.Linear(self.d_enc, d_model)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm_s = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout_s = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        tgt: Tensor,
        sent_memory: Tensor,
        graph_memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        sent_memory_mask: Optional[Tensor] = None,
        graph_memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        sent_memory_key_padding_mask: Optional[Tensor] = None,
        graph_memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
            Args:
            tgt (`FloatTensor`): set of `key_len`
                    key vectors `[batch, tgt_len, H]`
            memory (`FloatTensor`): set of `key_len`
                    key vectors `[batch, src_len, H]`
            tgt_mask: binary key2key mask indicating which keys have
                    non-zero attention `[tgt_len, tgt_len]`
            sent_memory_mask: binary key2key mask indicating which keys have
                    non-zero attention `[tgt_len, src_sent_len]`
            graph_memory_mask: binary key2key mask indicating which keys have
                    non-zero attention `[tgt_len, src_graph_len]`
            tgt_key_padding_mask: binary padding mask indicating which keys have
                    non-zero attention `[batch, tgt_len]`
            sent_memory_key_padding_mask: binary padding mask indicating which keys have
                    non-zero attention `[batch, src_sent_len]`
            graph_memory_key_padding_mask: binary padding mask indicating which keys have
                    non-zero attention `[batch, src_graph_len]`
            return:
            res:  [batch, tgt_len, H]
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        sent_tgt = self.sent_cross_attn(
            tgt, sent_memory, sent_memory, mask=sent_memory_mask, key_padding_mask=sent_memory_key_padding_mask)[0]
        tgt = tgt + self.dropout_s(sent_tgt)
        tgt = self.norm_s(tgt)
        if self.dual_enc:
            graph_tgt = self.graph_cross_attn(
                tgt, graph_memory, graph_memory, mask=graph_memory_mask, key_padding_mask=graph_memory_key_padding_mask)[0]
            tgt = tgt + self.dropout2(graph_tgt)
            tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class DoubleAttnTransformerDecoderLayerGraphFirst(nn.Module):
    def __init__(self, d_model, d_enc, heads, d_ff=2048, dropout=0.1, att_drop=0.1, activation="relu", dual_enc=True):
        super(DoubleAttnTransformerDecoderLayerGraphFirst, self).__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, query_dim=d_model, dropout=att_drop, use_structure=False)
        self.dual_enc = dual_enc
        self.d_enc = d_enc

        self.sent_cross_attn = MultiHeadedAttention(
            heads, d_model, query_dim=d_model, dropout=att_drop, use_structure=False
        )
        if dual_enc:
            self.graph_cross_attn = MultiHeadedAttention(
                heads, self.d_enc - d_model, query_dim=d_model, dropout=att_drop, use_structure=False
            )
            # self.fuse_linear = nn.Linear(self.d_enc, d_model)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm_g = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout_g = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        tgt: Tensor,
        sent_memory: Tensor,
        graph_memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        sent_memory_mask: Optional[Tensor] = None,
        graph_memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        sent_memory_key_padding_mask: Optional[Tensor] = None,
        graph_memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
            Args:
            tgt (`FloatTensor`): set of `key_len`
                    key vectors `[batch, tgt_len, H]`
            memory (`FloatTensor`): set of `key_len`
                    key vectors `[batch, src_len, H]`
            tgt_mask: binary key2key mask indicating which keys have
                    non-zero attention `[tgt_len, tgt_len]`
            sent_memory_mask: binary key2key mask indicating which keys have
                    non-zero attention `[tgt_len, src_sent_len]`
            graph_memory_mask: binary key2key mask indicating which keys have
                    non-zero attention `[tgt_len, src_graph_len]`
            tgt_key_padding_mask: binary padding mask indicating which keys have
                    non-zero attention `[batch, tgt_len]`
            sent_memory_key_padding_mask: binary padding mask indicating which keys have
                    non-zero attention `[batch, src_sent_len]`
            graph_memory_key_padding_mask: binary padding mask indicating which keys have
                    non-zero attention `[batch, src_graph_len]`
            return:
            res:  [batch, tgt_len, H]
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        if self.dual_enc:
            graph_tgt = self.graph_cross_attn(
                tgt, graph_memory, graph_memory, mask=graph_memory_mask, key_padding_mask=graph_memory_key_padding_mask)[0]
            tgt = tgt + self.dropout_g(graph_tgt)
            tgt = self.norm_g(tgt)
        
        tgt2 = self.sent_cross_attn(
            tgt, sent_memory, sent_memory, mask=sent_memory_mask, key_padding_mask=sent_memory_key_padding_mask)[0]
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

    def _check_args(self, src, lengths=None):
        _, n_batch = src.size()
        if lengths is not None:
            (n_batch_,) = lengths.size()
            # aeq(n_batch, n_batch_)

    def forward(self, src, src_key_padding_mask=None, mask=None, structure=None):
        """ See :obj:`EncoderBase.forward()`"""
        """
            Args:
            src (`FloatTensor`): set of vectors `[batch, seq_len, H]`
            mask: binary key2key mask indicating which keys have
                    non-zero attention `[batch, seq_len, seq_len]`
            src_key_padding_mask: binary key padding mask indicating which keys have
                    non-zero attention `[batch, 1, seq_len]`
            return:
            out_trans (`FloatTensor`): `[batch, seq_len, H]`

        """
        # self._check_args(src, lengths)
        out = src  # [B, seq_len, H]
        # Run the forward pass of every layer of the tranformer.
        for mod in self.layers:
            out = mod(out, mask, src_key_padding_mask, structure=structure)

        if self.norm is not None:
            out = self.norm(out)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt
        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class DoubleAttnTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(DoubleAttnTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        sent_memory: Tensor,
        graph_memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        sent_memory_mask: Optional[Tensor] = None,
        graph_memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        sent_memory_key_padding_mask: Optional[Tensor] = None,
        graph_memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt
        for mod in self.layers:
            output = mod(
                output,
                sent_memory,
                graph_memory,
                tgt_mask=tgt_mask,
                sent_memory_mask=sent_memory_mask,
                graph_memory_mask=graph_memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                sent_memory_key_padding_mask=sent_memory_key_padding_mask,
                graph_memory_key_padding_mask=graph_memory_key_padding_mask,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size, 1, seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()

    return (
        torch.arange(0, max_len, device=lengths.device)
        .type_as(lengths)
        .unsqueeze(0)
        .expand(batch_size, max_len)
        >= (lengths.unsqueeze(1))
    ).unsqueeze(1)


# class Transformer(nn.Module):
#     def __init__(self, args, embeddings):
#         super(Transformer, self).__init__()
#         self.args = args
#         self.in_dim = args.emb_dim + args.post_dim + args.pos_dim
#         if use_dep:
#             self.emb, self.pos_emb, self.post_emb, self.dep_emb = embeddings
#         else:
#             self.emb, self.pos_emb, self.post_emb = embeddings
