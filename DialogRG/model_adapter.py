import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nlp
from torch.distributions import Categorical
from nn_utils import PositionalEncoding, has_nan, universal_sentence_embedding, clip_and_normalize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from Transformer import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    DoubleAttnTransformerDecoder,
    DoubleAttnTransformerDecoderLayer,
    DoubleAttnTransformerDecoderLayerGraphFirst,
    DoubleAttnTransformerDecoderLayerSentFirst
)

cc = SmoothingFunction()

device = "cuda" if torch.cuda.is_available() else "cpu"


class BeamInstance:
    def __init__(self, ids, neg_logp, is_finish):
        self.ids = ids
        self.neg_logp = neg_logp
        self.is_finish = is_finish

    def get_logp_norm(self, eos_id):
        try:
            l = self.ids.index(eos_id)
            return self.neg_logp / (l)
        except ValueError:
            return self.neg_logp / (len(self.ids) - 1)

    def get_ids(self, eos_id):
        try:
            i = self.ids.index(eos_id)
            return self.ids[1:i]
        except ValueError:
            return self.ids[1:]


class SentTransformer(nn.Module):
    def __init__(self, config, bpemb, vocab):
        super(SentTransformer, self).__init__()

        self.config = config
        self.embedding = bpemb
        self.vocab = vocab

        assert vocab is not None
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.position_encoder = PositionalEncoding(config.d_enc_sent)
        # encoder_layer = nn.TransformerEncoderLayer(
        #     config.d_model, config.n_head, dim_feedforward=1024, dropout=config.dropout
        # )
        encoder_layer = TransformerEncoderLayer(
            d_model=config.d_enc_sent,
            heads=config.n_head,
            d_ff=getattr(config, "d_ff", 1024),
            dropout=config.dropout,
            att_drop=config.dropout,
            use_structure=False,
        )
        encoder_norm = nn.LayerNorm(config.d_enc_sent)
        self.encoder = TransformerEncoder(encoder_layer, config.num_layer, encoder_norm)

        if vocab is not None:
            self.vocab_size = len(self.vocab)
            self.BOS = self.vocab["<bos>"]
            self.EOS = self.vocab["<eos>"]
        else:
            self.vocab_size = self.bpemb.vectors.shape[0]
            self.BOS = self.bpemb.BOS
            self.EOS = self.bpemb.EOS

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch):
        # encoding
        src_mask_inv = (batch["src_mask"] == 0).to(device)   # [batch, seq]
        # batch["src_mask_inv"] = src_mask_inv
        src_emb = self.embedding(batch["src"].to(device))  # [batch, seq, dim]
        # print("input_size", history_emb.size())
        src_emb = self.position_encoder(src_emb.transpose(0, 1)).transpose(0, 1)                                       # [batch, seq, dim]
        src_emb = self.encoder(src_emb, src_key_padding_mask=src_mask_inv)  # [batch, seq, dim]
        assert has_nan(src_emb) is False
        return src_emb


class GraphTransformer(nn.Module):
    def __init__(self, config, bpemb, vocab, relation_vocab):
        super(GraphTransformer, self).__init__()

        self.config = config
        self.embedding = bpemb
        self.vocab = vocab
        self.relation_vocab = relation_vocab
        self.n_layer = getattr(config, "g_num_layer", 4)
        self.use_pe = getattr(config, "g_pe", True)

        assert vocab is not None
        self.vocab_inv = {v: k for k, v in vocab.items()}
        assert relation_vocab is not None
        self.relation_embedding = nn.Embedding(len(self.relation_vocab), config.d_relation)
        assert config.d_relation * config.g_n_head == config.d_enc_concept

        self.position_encoder = PositionalEncoding(config.d_enc_concept)

        # encoder_layer = nn.TransformerEncoderLayer(
        #     config.d_model, config.n_head, dim_feedforward=1024, dropout=config.dropout
        # )
        encoder_layer = TransformerEncoderLayer(
            d_model=config.d_enc_concept,
            heads=config.g_n_head,
            d_ff=getattr(config, "g_d_ff", 1024),
            dropout=config.dropout,
            att_drop=config.dropout,
            use_structure=True,
        )
        encoder_norm = nn.LayerNorm(config.d_enc_concept)
        self.encoder = TransformerEncoder(encoder_layer, self.n_layer, encoder_norm)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch):
        # encoding
        src_mask_inv = (batch["con_mask"] == 0).to(device)  # [batch, seq]
        # batch["con_mask_inv"] = src_mask_inv
        src_emb = self.embedding(batch["con"].to(device))  # [batch, seq, dim]
        structure_emb = self.relation_embedding(batch["rel"].to(device))  # [batch, seq, seq, s_dim]
        # print('structure size:', structure_emb.size())
        # print("input_size", src_emb.size())
        if self.use_pe:
            src_emb = self.position_encoder(src_emb.transpose(0, 1)).transpose(0, 1)  # [batch, seq, dim]
        src_emb = self.encoder(
            src_emb, src_key_padding_mask=src_mask_inv, structure=structure_emb
        )  # [batch, seq, dim]
        # batch["con_emb"] = src_emb
        assert has_nan(src_emb) is False
        return src_emb


class AdapterGraphTransformer(nn.Module):
    def __init__(self, config, relation_vocab):
        super(AdapterGraphTransformer, self).__init__()

        self.config = config
        self.relation_vocab = relation_vocab
        self.n_layer = getattr(config, "adapter_layer", 2)
        self.use_pe = getattr(config, "adapter_pe", True)

        assert relation_vocab is not None
        d_relation = config.d_enc_sent // config.n_head
        self.relation_embedding = nn.Embedding(len(self.relation_vocab), d_relation)
        self.position_encoder = PositionalEncoding(config.d_enc_sent)

        encoder_layer = TransformerEncoderLayer(
            d_model=config.d_enc_sent,
            heads=config.n_head,
            d_ff=getattr(config, "g_d_ff", 1024),
            dropout=config.dropout,
            att_drop=config.dropout,
            use_structure=True,
        )
        encoder_norm = nn.LayerNorm(config.d_enc_sent)
        self.encoder = TransformerEncoder(encoder_layer, self.n_layer, encoder_norm)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_emb, batch):
        # encoding
        src_mask_inv = (batch["src_mask"] == 0).to(device)   # [batch, seq]
        # src_mask_inv = batch["src_mask_inv"]
        # print('src_emb:', src_emb.size())
        # assert len(src_emb.size()) == 3, 'Invalid input size:{}, should be 3!!'.format(' x '.join([str(itm) for itm in src_emb.size()]))
        structure_emb = self.relation_embedding(batch["wr"].to(device))  # [batch, seq, seq, s_dim]
        assert len(structure_emb.size()) == 4, 'Invalid input size:{}, should be 3!!'.format(' x '.join([str(itm) for itm in structure_emb.size()]))
        assert src_emb.size(1) == structure_emb.size(1) and src_emb.size(1) == structure_emb.size(2)
        # print('structure size:', structure_emb.size())
        # print("input_size", src_emb.size())
        if self.use_pe:
            src_emb = self.position_encoder(src_emb.transpose(0, 1)).transpose(0, 1)  # [batch, seq, dim]
        # print('src_emb_pos:', src_emb.size())
        src_emb = self.encoder(
            src_emb, src_key_padding_mask=src_mask_inv, structure=structure_emb
        )  # [batch, seq, dim]
        # exit()
        assert has_nan(src_emb) is False
        return src_emb


class DualTransformer(nn.Module):
    def __init__(self, config, word_emb, con_emb=None, word_vocab=None, concept_vocab=None, relation_vocab=None, word_rel_vocab=None):
        super(DualTransformer, self).__init__()

        self.config = config
        self.word_vocab = word_vocab
        self.concept_vocab = concept_vocab
        self.relation_vocab = relation_vocab

        self.enc_word_embedding = self.build_embedding(word_emb, word_vocab, self.config.d_enc_sent)
        self.word_encoder = SentTransformer(config, self.enc_word_embedding, word_vocab)
        if config.dual_enc and self.concept_vocab is not None and relation_vocab is not None:
            if config.share_con_vocab:
                self.enc_concept_embedding = self.enc_word_embedding
            else:
                self.enc_concept_embedding = self.build_embedding(con_emb, concept_vocab, self.config.d_enc_concept)
            
            self.graph_encoder = GraphTransformer(config, self.enc_concept_embedding, concept_vocab, relation_vocab)
        else:
            self.graph_encoder = None
        if config.use_adapter and word_rel_vocab is not None:
            self.adapter_enc = AdapterGraphTransformer(config, word_rel_vocab)
            self.adapter_norm = nn.LayerNorm(config.d_enc_sent)
        else:
            self.adapter_enc = None

        self.dec_word_embedding = self.enc_word_embedding
        self.position_encoder = PositionalEncoding(config.d_dec)
        dual_mode = getattr(config, "dual_mode", "cat")
        if config.dual_enc:
            if dual_mode == "cat":
                decoder_layer = DoubleAttnTransformerDecoderLayer(
                    d_model=config.d_dec,
                    d_sent=config.d_enc_sent,
                    d_con=config.d_enc_concept,
                    heads=config.n_head,
                    d_ff=1024,
                    dropout=config.dropout,
                    att_drop=config.dropout,
                    dual_enc=config.dual_enc,                               # dual_enc=False when use single sentence encoder
                )
            # elif dual_mode == "graph_first":
            #     decoder_layer = DoubleAttnTransformerDecoderLayerGraphFirst(
            #         d_model=config.d_model,
            #         d_enc=config.d_model + config.d_concept if config.dual_enc else config.d_model,
            #         heads=config.n_head,
            #         d_ff=1024,
            #         dropout=config.dropout,
            #         att_drop=config.dropout,
            #         dual_enc=config.dual_enc,                             # dual_enc=False when use single sentence encoder
            #     )
            # elif dual_mode == "sent_first":
            #     decoder_layer = DoubleAttnTransformerDecoderLayerSentFirst(
            #         d_model=config.d_model,
            #         d_enc=config.d_model + config.d_concept if config.dual_enc else config.d_model,
            #         heads=config.n_head,
            #         d_ff=1024,
            #         dropout=config.dropout,
            #         att_drop=config.dropout,
            #         dual_enc=config.dual_enc,                             # dual_enc=False when use single sentence encoder
            #     )
            else:
                print('Invalid dual_mode, should in (cat, graph_first, sent_first)')
        else:
            decoder_layer = DoubleAttnTransformerDecoderLayer(
                d_model=config.d_dec,
                d_sent=config.d_enc_sent,
                d_con=config.d_enc_concept,
                heads=config.n_head,
                d_ff=1024,
                dropout=config.dropout,
                att_drop=config.dropout,
                dual_enc=config.dual_enc,                                # dual_enc=False when use single sentence encoder
            )
        decoder_norm = nn.LayerNorm(config.d_dec)
        self.decoder = DoubleAttnTransformerDecoder(decoder_layer, config.num_layer, decoder_norm)

        if word_vocab is not None:
            self.word_vocab_size = len(self.word_vocab)
            self.BOS = self.word_vocab["<bos>"]
            self.EOS = self.word_vocab["<eos>"]

        self.projector = nn.Linear(config.d_dec, self.word_vocab_size)
        if self.config.share_vocab:             # existing bugs to be fixed
            self.projector.weight = self.dec_word_embedding.weight
        if self.config.use_kl_loss:
            self.kl = nn.KLDivLoss(size_average=False)

        if self.config.rl_ratio > 0.0 and self.config.rl_type == "bertscore":
            self.rl_metric = nlp.load_metric("bertscore")

    def decode_into_string(self, ids):
        try:
            i = ids.index(self.EOS)
            ids = ids[:i]
        except ValueError:
            pass
        if self.word_vocab is not None:
            return " ".join([self.word_encoder.vocab_inv[x] for x in ids])
        # else:
        #     return self.bpemb.decode_ids(ids)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def build_embedding(self, pretrain_emb, vocab, d_emb):
        freeze_emb = getattr(self.config, "freeze_emb", True)
        if pretrain_emb is not None:
            if pretrain_emb.shape[1] == d_emb:
                embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrain_emb).to(device), freeze=freeze_emb)
            else: 
                embedding = nn.Sequential(
                    nn.Embedding.from_pretrained(torch.from_numpy(pretrain_emb).to(device), freeze=freeze_emb),
                    nn.Linear(pretrain_emb.shape[1], d_emb),
                )
        else:
            embedding = nn.Embedding(len(vocab), d_emb)
        return embedding

    # greey or sampling based inference
    def inference(
        self,
        sent_memory_emb,
        graph_memory_emb,
        sent_memory_mask,
        graph_memory_mask,
        max_step,
        use_sampling=False,
    ):
        batch_size, sent_memory_seq, dim = list(sent_memory_emb.shape)
        _, graph_memory_seq, _ = list(graph_memory_emb.shape)

        sent_memory_mask_inv = sent_memory_mask == 0    # [batch, sent_memory_seq]
        graph_memory_mask_inv = graph_memory_mask == 0  # [batch, sent_memory_seq]

        target_ids = [[self.BOS for i in range(batch_size)]]    # [target_seq, batch]
        target_mask = [[1.0] for i in range(batch_size)]        # [batch, target_seq]
        target_prob = []                                        # [target_seq, batch]
        is_finish = [False for _ in range(batch_size)]
        rows = torch.arange(batch_size).to(device)
        for step in range(max_step):
            cur_seq = step + 1
            cur_emb = self.dec_word_embedding(torch.tensor(target_ids).to(device))  # [cur_seq, batch, dim]
            cur_emb = self.position_encoder(cur_emb)    # [cur_seq, batch, dim]

            cur_mask = torch.tensor(target_mask).to(device)
            cur_mask_inv = cur_mask == 0.0              # [batch, cur_seq]
            cur_triu_mask = torch.triu(
                torch.ones(cur_seq, cur_seq).to(device), diagonal=1
            )  # [cur_seq, cur_seq]
            cur_triu_mask.masked_fill_(cur_triu_mask == 1, -1e20)

            cur_emb = self.decoder(
                cur_emb,
                sent_memory_emb,  # [batch, sent_len, dim]
                graph_memory_emb,  # [batch, graph_len, dim]
                tgt_mask=cur_triu_mask,
                tgt_key_padding_mask=cur_mask_inv,
                sent_memory_key_padding_mask=sent_memory_mask_inv,
                graph_memory_key_padding_mask=graph_memory_mask_inv,
            )  # [batch, cur_seq, dim]

            assert has_nan(cur_emb) is False

            # break after the first time when all items are finished
            if all(is_finish) or step == max_step - 1:
                cur_len = cur_mask.sum(dim=1).long()
                target_vec = universal_sentence_embedding(cur_emb, cur_mask, cur_len)
                break

            # generating step outputs
            logits = self.projector(cur_emb[:, -1, :]).view(
                batch_size, self.word_vocab_size
            )  # [batch, vocab]
            if use_sampling is False:
                indices = logits.argmax(dim=1)                  # [batch]
            else:
                indices = Categorical(logits=logits).sample()   # [batch]

            prob = F.softmax(logits, dim=1)[rows, indices]      # [batch]
            target_prob.append(prob)
            indices = indices.cpu().tolist()
            target_ids.append(indices)
            for i in range(batch_size):
                target_mask[i].append(
                    0.0 if is_finish[i] else 1.0
                )  # based on if is_finish in the last step

            for i in range(batch_size):
                is_finish[i] |= indices[i] == self.EOS

        target_ids = list(map(list, zip(*target_ids[1:])))  # [batch, target_seq]
        target_mask = torch.tensor([x[1:] for x in target_mask]).to(device)  # [batch, target_seq]
        target_prob = torch.stack(target_prob, dim=1)  # [batch, target_seq]
        return target_vec, target_ids, target_prob, target_mask

    def forward(self, batch):
        # encoding
        sent_mask_inv = (batch["src_mask"] == 0).to(device)  # [batch, seq]
        graph_mask_inv = (batch["con_mask"] == 0).to(device)
        sent_mem = self.word_encoder(batch)
        if self.adapter_enc is not None:
            sent_mem_new = self.adapter_enc(sent_mem, batch)
            sent_mem = self.adapter_norm(sent_mem_new + sent_mem)
        graph_mem = self.graph_encoder(batch) if self.graph_encoder is not None else None
        
        # decoding, batch['target'] includes both <bos> and <eos>
        if batch["tgt_ref"] is not None:
            target_input = batch["tgt_input"].to(device)   # [batch, trg_seq]
            target_ref = batch["tgt_ref"].to(device)       # [batch, trg_seq]
            bsz, trg_seq = target_input.size()
            triangle_mask = torch.triu(
                torch.ones(trg_seq, trg_seq).to(device), diagonal=1
            )  # [trg_seq, trg_seq]
            triangle_mask.masked_fill_(triangle_mask == 1, -1e20)
            triangle_mask = triangle_mask.repeat(bsz, 1, 1)
            target_mask_inv = (batch["tgt_mask"] == 0).to(device)  # [batch, trg_seq]

            target_emb = self.dec_word_embedding(target_input).transpose(0, 1)  # [batch, trg_seq, dim]
            target_emb = self.position_encoder(target_emb).transpose(0, 1)  # [batch, trg_seq, dim]
            # print('tgt_emb_size:', target_emb.size())
            # print('memory_size:', combined_emb.size())
            target_emb = self.decoder(
                target_emb,
                sent_mem,
                graph_mem,
                tgt_mask=triangle_mask,
                tgt_key_padding_mask=target_mask_inv,
                sent_memory_key_padding_mask=sent_mask_inv,
                graph_memory_key_padding_mask=graph_mask_inv,
            )  # [batch, trg_seq, dim]

            assert has_nan(target_emb) is False
            # generating outputs
            logits = self.projector(target_emb)  # [batch, trg_seq, vocab]
            preds = logits.argmax(dim=2)  # [batch, trg_seq]
            loss = F.cross_entropy(
                logits.contiguous().view(-1, self.word_vocab_size),
                target_ref.contiguous().view(-1),
                ignore_index=0,
            )
            train_right = ((preds == target_ref).float() * batch["tgt_mask"].to(device)).sum()
            train_total = batch["tgt_mask"].to(device).sum()

            return {
                "preds": preds,
                "loss": loss,
                "counts": (train_right, train_total),
                "selected_kn": None,
                "trg_selected_kn": None,
            }
        else:
            return {
                "sent_memory_emb": sent_mem,
                "sent_memory_mask": batch["src_mask"],
                "graph_memory_emb": graph_mem,
                "graph_memory_mask": batch["con_mask"],
                "selected_kn": None,
                "trg_selected_kn": None,
            }

    def decode(
        self,
        sent_memory_emb,
        graph_memory_emb,
        sent_memory_mask,
        graph_memory_mask,
        beamsize,
        max_step,
    ):  # [batch, seq, dim]
        batch_size, sent_memory_seq, s_dim = list(sent_memory_emb.shape)
        if graph_memory_emb is not None:
            _, graph_memory_seq, g_dim = list(graph_memory_emb.shape)
        beam = [
            [BeamInstance(ids=[self.BOS], neg_logp=0.0, is_finish=False)] for i in range(batch_size)
        ]
        cur_beamsize = 1
        for step in range(max_step):
            cur_seq = step + 1
            target_input = [
                [beam[i][j].ids for j in range(cur_beamsize)] for i in range(batch_size)
            ]   # [batch, beam, cur_seq]
            target_input = (
                torch.tensor(target_input).to(device).view(batch_size * cur_beamsize, cur_seq)
            )   # [batch*beam, cur_seq]
            target_emb = self.dec_word_embedding(target_input).transpose(0, 1)  # [cur_seq, batch*beam, dim]
            target_emb = self.position_encoder(target_emb).transpose(
                0, 1
            )   # [batch*beam, cur_seq, dim]

            cur_sent_memory_emb = (
                sent_memory_emb.unsqueeze(dim=1)
                .repeat(1, cur_beamsize, 1, 1)
                .view(batch_size * cur_beamsize, sent_memory_seq, s_dim)
            )   # [batch*beam, sent_memory_seq, dim]
            cur_sent_memory_mask_inv = (
                sent_memory_mask.unsqueeze(dim=1)
                .repeat(1, cur_beamsize, 1)
                .view(batch_size * cur_beamsize, sent_memory_seq)
                == 0
            )  # [batch*beam, graph_memory_seq]
            if graph_memory_emb is not None:
                cur_graph_memory_emb = (
                    graph_memory_emb.unsqueeze(dim=1)
                    .repeat(1, cur_beamsize, 1, 1)
                    .view(batch_size * cur_beamsize, graph_memory_seq, g_dim)
                )  # [batch*beam, graph_memory_seq, dim]
                cur_graph_memory_mask_inv = (
                    graph_memory_mask.unsqueeze(dim=1)
                    .repeat(1, cur_beamsize, 1)
                    .view(batch_size * cur_beamsize, graph_memory_seq)
                    == 0
                )  # [batch*beam, graph_memory_seq]
            else:
                cur_graph_memory_emb = None
                cur_graph_memory_mask_inv = None
            cur_triu_mask = torch.triu(
                torch.ones(cur_seq, cur_seq).to(device), diagonal=1
            )  # [cur_seq, cur_seq]
            cur_triu_mask = cur_triu_mask.repeat(batch_size * cur_beamsize, 1, 1)
            cur_triu_mask.masked_fill_(cur_triu_mask == 1, -1e20)

            target_emb = self.decoder(
                target_emb,
                cur_sent_memory_emb,
                cur_graph_memory_emb,
                tgt_mask=cur_triu_mask,
                tgt_key_padding_mask=None,
                sent_memory_key_padding_mask=cur_sent_memory_mask_inv,
                graph_memory_key_padding_mask=cur_graph_memory_mask_inv,
            )  # [batch*beam, cur_seq, dim]

            assert has_nan(target_emb) is False

            # generating step outputs
            logits = self.projector(target_emb[:, -1, :]).view(
                batch_size, cur_beamsize, self.word_vocab_size
            )  # [batch, beam, vocab]
            logits = F.log_softmax(logits, dim=2)
            indices = logits.topk(beamsize + 1, dim=2)[1].cpu().numpy()  # [batch, beam, topk]

            all_finish = True
            next_beam = []
            for i in range(batch_size):
                cands = []
                for j in range(cur_beamsize):
                    if beam[i][j].is_finish:
                        cands.append(
                            (0, beam[i][j].neg_logp, self.EOS, j)
                        )  # to make sure 'finished' are in the front
                    else:
                        for nid in indices[i, j]:
                            neg_logp = beam[i][j].neg_logp - logits[i, j, nid].item()
                            cands.append((1, neg_logp, int(nid), j))  # '0' finished; '1' unfinished
                assert len(cands) >= beamsize
                assert sum([x[0] == 0 for x in cands]) <= beamsize, cands
                cands.sort()

                next_beam.append([])
                for _, neg_logp, nid, j in cands[:beamsize]:
                    is_finish = beam[i][j].is_finish or nid == self.EOS
                    all_finish &= is_finish
                    next_instance = BeamInstance(
                        ids=beam[i][j].ids + [nid, ], neg_logp=neg_logp, is_finish=is_finish
                    )
                    next_beam[-1].append(next_instance)

            # preparing for the next loop
            if all_finish:
                break

            beam = next_beam
            cur_beamsize = beamsize

        best = []
        for i in range(batch_size):
            indices = np.argsort([x.get_logp_norm(self.EOS) for x in beam[i]])
            j = indices[0]
            best.append(beam[i][j].get_ids(self.EOS))
        return best