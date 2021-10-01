# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" PyTorch T5 model. """


import copy
import logging
import math
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.file_utils import DUMMY_INPUTS, DUMMY_MASK, add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_t5 import T5Attention, T5LayerNorm, T5DenseReluDense, T5LayerFF, load_tf_weights_in_t5, \
    T5_PRETRAINED_MODEL_ARCHIVE_MAP, T5LayerSelfAttention, T5LayerCrossAttention, \
    T5_START_DOCSTRING, T5_INPUTS_DOCSTRING

from .configuration_auxseq import T5Config
from .modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)

####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of torch.nn.Module)
####################################################

class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config, has_relative_attention_bias=has_relative_attention_bias))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        head_mask=None,
        past_key_value_state=None,
        use_cache=False,
    ):

        if past_key_value_state is not None:
            assert self.is_decoder, "Only decoder can use `past_key_value_states`"
            expected_num_past_key_value_states = 2 if encoder_hidden_states is None else 4

            error_message = "There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states".format(
                expected_num_past_key_value_states,
                "2 (past / key) for cross attention" if expected_num_past_key_value_states == 4 else "",
                len(past_key_value_state),
            )
            assert len(past_key_value_state) == expected_num_past_key_value_states, error_message

            self_attn_past_key_value_state = past_key_value_state[:2]
            cross_attn_past_key_value_state = past_key_value_state[2:]
        else:
            self_attn_past_key_value_state, cross_attn_past_key_value_state = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value_state=self_attn_past_key_value_state,
            use_cache=use_cache,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        self.self_attention_outputs = hidden_states
        # _, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        if self.is_decoder and encoder_hidden_states is not None:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                kv=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                head_mask=head_mask,
                past_key_value_state=cross_attn_past_key_value_state,
                query_length=query_length,
                use_cache=use_cache,
            )
            hidden_states = cross_attention_outputs[0]
            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)
        outputs = (hidden_states,)

        # Add attentions if we output them
        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs  # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)


class T5PreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
        NOTE: This is the same as the T5PreTrainedModel from Huggingface. But it inherits from our PreTrainedModel.
    """

    config_class = T5Config
    pretrained_model_archive_map = T5_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = "transformer"

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """ Initialize the weights """
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        # elif isinstance(module, (T5ForConditionalGenerationDualEmb, T5ForConditionalGenerationDualEmbCntAction)):
        #     # Mesh TensorFlow embeddings initialization
        #     # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
        #     module.src_word_emb_func.weight.data.normal_(mean=0.0, std=factor * 1.0)
        #     module.src_word_emb_prim.weight.data.normal_(mean=0.0, std=factor * 1.0)
        #     module.trg_word_emb.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            d_kv = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * d_kv) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * d_kv) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in lm_labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `lm_labels` has only positive values and -100"

        return shifted_input_ids


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None, embedding_regularization=False, layer_regularization=False, embed_positions=None,
                 embed_action_count=None, embed_action_group=None):
        super().__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.mask_decoder_input = config.mask_decoder_input

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.embedding_regularization = embedding_regularization
        self.layer_regularization = layer_regularization
        self.embed_positions = embed_positions
        self.embed_action_count = embed_action_count
        self.embed_action_group = embed_action_group

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        self.noise_weight = 1.0
        self.use_l1_norm = False
        self.sample_wise_content_noise = True

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def noise_regularization(self, rep, mask):
        bsz, qlen = rep.size(0), rep.size(1)
        # if self.content_noise_coe > 0:
        if self.use_l1_norm:
            norm = torch.mean(torch.abs(rep), dim=- 1)
        else:
            norm = torch.mean(rep ** 2, dim=-1)

        masked_norm = norm * mask
        if self.sample_wise_content_noise:
            noise_reg = torch.sum(masked_norm / bsz)
        else:
            norm_sum = torch.sum(masked_norm, -1)
            noise_reg_sample = norm_sum / qlen
            noise_reg = torch.mean(noise_reg_sample)
            #self.regularization_list.append(self.reg_coe * noise_reg)
        if self.training:
            noisy_rep = rep + self.noise_weight * torch.normal(mean=0, std=1.0, size=rep.size()).cuda()
        else:
            noisy_rep = rep
        return noisy_rep, noise_reg

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        action_count_ids=None,
        action_group_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        past_key_value_states=None,
        use_cache=False,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            if self.is_decoder:
                raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)


            if self.embed_positions is not None:
                if position_ids is None:
                    position_embeds = self.embed_positions(torch.arange(seq_length).cuda()).unsqueeze(0)
                else:
                    position_embeds = self.embed_positions(position_ids).unsqueeze(0)
                inputs_embeds = inputs_embeds + position_embeds

            if self.embed_action_count is not None:
            #     assert action_count_ids is not None
                action_count_embeds = self.embed_action_count(action_count_ids)
            #     inputs_embeds = inputs_embeds + action_count_embeds

            if self.embed_action_group is not None:
            #     assert action_group_ids is not None
                action_group_embeds = self.embed_action_group(action_group_ids)
            #     inputs_embeds = inputs_embeds + action_group_embeds

        if past_key_value_states is not None:
            assert seq_length == 1, "Input shape is {}, but should be {} when using past_key_value_sates".format(
                input_shape, (batch_size, 1)
            )
            # required mask seq length can be calculated via length of past
            # key value states and seq_length = 1 for the last token
            mask_seq_length = past_key_value_states[0][0].shape[2] + seq_length
        else:
            mask_seq_length = seq_length

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)

        if self.embedding_regularization:
            inputs_embeds, self.func_emb_reg_loss = self.noise_regularization(inputs_embeds, attention_mask)
            # if self.embed_action_count is not None:
            #     action_count_embeds, self.action_count_emb_reg_loss = self.noise_regularization(action_count_embeds, attention_mask)
            #     self.func_emb_reg_loss = self.func_emb_reg_loss + action_count_emb_reg_loss
            # if self.embed_action_group is not None:
            #     action_group_embeds, self.action_group_emb_reg_loss = self.noise_regularization(action_group_embeds, attention_mask)
            #     self.func_emb_reg_loss = self.func_emb_reg_loss + action_group_emb_reg_loss

        # if self.is_decoder and self.mask_decoder_input:
        #     inputs_embeds = torch.zeros_like(inputs_embeds).cuda()

        if self.embed_action_count:
            inputs_embeds = inputs_embeds + action_count_embeds
        if self.embed_action_group:
            inputs_embeds = inputs_embeds + action_group_embeds

        if self.is_decoder and self.mask_decoder_input:
            inputs_embeds = torch.zeros_like(inputs_embeds).cuda()

        # if self.is_decoder and self.mask_decoder_input and self.training:
        #     inputs_zero_embeds = torch.zeros_like(inputs_embeds).cuda()
        #     rand_mask = torch.rand(batch_size).unsqueeze(1).unsqueeze(1).cuda()
        #     inputs_embeds = torch.where(rand_mask < 0.5, inputs_embeds, inputs_zero_embeds)

        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_length).to(inputs_embeds.device)

        # initialize past_key_value_states with `None` if past does not exist
        if past_key_value_states is None:
            past_key_value_states = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, self.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = ()
        all_hidden_states = ()
        all_attentions = ()
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        layer_reg_losses = []

        for i, (layer_module, past_key_value_state) in enumerate(zip(self.block, past_key_value_states)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask[i],
                past_key_value_state=past_key_value_state,
                use_cache=use_cache,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]
            if self.layer_regularization:
                hidden_states, layer_reg_loss = self.noise_regularization(hidden_states, attention_mask)
                layer_reg_losses.append(layer_reg_loss.unsqueeze(0))

            if i == 0:
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[3 if self.output_attentions else 2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[5 if self.output_attentions else 3]
                self.first_layer_output = hidden_states
            # append next layer key value states
            present_key_value_states = present_key_value_states + (present_key_value_state,)

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.layer_regularization:
            self.layer_reg_losses = torch.sum(torch.cat(layer_reg_losses))

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if use_cache is True:
            assert self.is_decoder, "`use_cache` can only be set to `True` if {} is used as a decoder".format(self)
            outputs = outputs + (present_key_value_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (presents,) (all hidden states), (all attentions)


class MHAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        # self.output_attentions = config.output_attentions
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.d_kv

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        # self.pruned_heads = set()

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=not self.is_decoder,
            num_buckets=self.relative_attention_num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)
        return values

    def forward(
        self,
        input,
        mask=None,
        k=None,
        v=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
    ):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        # past_key_value_state[0] is (bs, n_heads, q_len - 1, dim_per_head)
        bs, qlen, dim = input.size()

        if past_key_value_state is not None:
            # assert self.is_decoder is True, "Encoder cannot cache past key value states"
            assert (
                len(past_key_value_state) == 2
            ), "past_key_value_state should have 2 past states: keys and values. Got {} past states".format(
                len(past_key_value_state)
            )
            real_qlen = qlen + past_key_value_state[0].shape[2] if query_length is None else query_length
        else:
            real_qlen = qlen

        if k is None:
            klen = real_qlen
        else:
            klen = k.size(1)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

        q = shape(self.q(input))  # (bs, n_heads, qlen, dim_per_head)

        if k is None:
            assert False
            k = shape(self.k(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(input))  # (bs, n_heads, qlen, dim_per_head)
        elif past_key_value_state is None:
            # k = v = kv
            k = shape(self.k(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(v))  # (bs, n_heads, qlen, dim_per_head)

        if past_key_value_state is not None:
            assert False
            if k is None:
                k_, v_ = past_key_value_state
                k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
            else:
                k, v = past_key_value_state

        # if self.is_decoder and use_cache is True:
        #     present_key_value_state = ((k, v),)
        # else:
        #     present_key_value_state = (None,)

        scores = torch.einsum("bnqd,bnkd->bnqk", q, k)  # (bs, n_heads, qlen, klen)

        if position_bias is None:
            if not self.has_relative_attention_bias:
                raise ValueError("No position_bias provided and no weights to compute position_bias")
            position_bias = self.compute_bias(real_qlen, klen)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value_state is not None:
                position_bias = position_bias[:, :, -1:, :]

            if mask is not None:
                position_bias = position_bias + mask  # (bs, n_heads, qlen, klen)

        scores += position_bias
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)
        # print(weights[0, 0])
        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)
        self.filler = context
        context = self.o(context)

        outputs = (context,) # + present_key_value_state

        # if self.output_attentions:
        #     outputs = outputs + (weights,)
        # if self.has_relative_attention_bias:
        #     outputs = outputs + (position_bias,)
        return outputs


class T5StackWithDualEmb(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None, embed_tokens_prim=None, embedding_regularization=False, layer_regularization=False):
        super().__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.embed_tokens = embed_tokens
        self.embed_tokens_prim = embed_tokens_prim
        self.is_decoder = config.is_decoder
        self.mask_decoder_input = config.mask_decoder_input
        self.embedding_regularization = embedding_regularization
        self.layer_regularization = layer_regularization

        if self.is_decoder:
            self.vocab_size = config.trg_vocab_size
        else:
            self.vocab_size = config.vocab_size

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # self.content_noise_coe = 0.1
        self.noise_weight = 1.0
        self.use_l1_norm = False
        self.sample_wise_content_noise = True

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def noise_regularization(self, rep, mask):
        bsz, qlen = rep.size(0), rep.size(1)
        # if self.content_noise_coe > 0:
        if self.use_l1_norm:
            norm = torch.mean(torch.abs(rep), dim=- 1)
        else:
            norm = torch.mean(rep ** 2, dim=-1)

        masked_norm = norm * mask
        if self.sample_wise_content_noise:
            noise_reg = torch.sum(masked_norm / bsz)
        else:
            norm_sum = torch.sum(masked_norm, -1)
            noise_reg_sample = norm_sum / qlen
            noise_reg = torch.mean(noise_reg_sample)
            #self.regularization_list.append(self.reg_coe * noise_reg)
        if self.training:
            noisy_rep = rep + self.noise_weight * torch.normal(mean=0, std=1.0, size=rep.size()).cuda()
        else:
            noisy_rep = rep
        return noisy_rep, noise_reg

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        past_key_value_states=None,
        use_cache=False,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            if self.is_decoder:
                raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            assert self.embed_tokens_prim is not None
            inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds_prim = self.embed_tokens_prim(input_ids)
            if self.embedding_regularization and not self.is_decoder:
                inputs_embeds, self.func_emb_reg_loss = self.noise_regularization(inputs_embeds, attention_mask)
                inputs_embeds_prim, self.prim_emb_reg_loss = self.noise_regularization(inputs_embeds_prim, attention_mask)

                # diag_input_ids = torch.diag_embed(torch.ones(self.vocab_size))
                #  = self.noise_regularization_loss(self.embed_tokens(diag_input_ids), encoder_attention_mask)
                #  = self.noise_regularization_loss(self.embed_tokens_prim(diag_input_ids), encoder_attention_mask)

            if self.is_decoder and self.mask_decoder_input:
                inputs_embeds = torch.zeros_like(inputs_embeds).cuda()

        batch_size, seq_length = input_shape

        if past_key_value_states is not None:
            assert seq_length == 1, "Input shape is {}, but should be {} when using past_key_value_sates".format(
                input_shape, (batch_size, 1)
            )
            # required mask seq length can be calculated via length of past
            # key value states and seq_length = 1 for the last token
            mask_seq_length = past_key_value_states[0][0].shape[2] + seq_length
        else:
            mask_seq_length = seq_length

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_length).to(inputs_embeds.device)

        # initialize past_key_value_states with `None` if past does not exist
        if past_key_value_states is None:
            past_key_value_states = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, self.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = ()
        all_hidden_states = ()
        all_attentions = ()
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        layer_reg_losses = []

        for i, (layer_module, past_key_value_state) in enumerate(zip(self.block, past_key_value_states)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask[i],
                past_key_value_state=past_key_value_state,
                use_cache=use_cache,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]
            if self.layer_regularization:
                hidden_states, layer_reg_loss = self.noise_regularization(hidden_states, attention_mask)
                layer_reg_losses.append(layer_reg_loss.unsqueeze(0))

            if i == 0:
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[3 if self.output_attentions else 2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[5 if self.output_attentions else 3]
                self.first_layer_output = hidden_states

            # append next layer key value states
            present_key_value_states = present_key_value_states + (present_key_value_state,)

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if self.layer_regularization:
            self.layer_reg_losses = torch.sum(torch.cat(layer_reg_losses))

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states, )
        if not self.is_decoder:
            outputs = outputs + (inputs_embeds_prim, inputs_embeds)
        if use_cache is True:
            assert self.is_decoder, "`use_cache` can only be set to `True` if {} is used as a decoder".format(self)
            outputs = outputs + (present_key_value_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (presents,) (all hidden states), (all attentions)


@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)

class T5ForConditionalGenerationWithSepVocab(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model
        self.predict_action_count = config.predict_action_count

        self.src_word_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.trg_word_emb = nn.Embedding(config.trg_vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        self.encoder = T5Stack(encoder_config, self.src_word_emb)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = T5Stack(decoder_config, self.trg_word_emb)

        self.lm_head = nn.Linear(config.d_model, config.trg_vocab_size, bias=False)
        self.trg_vocab_size = config.trg_vocab_size
        if self.predict_action_count:
            self.action_count_head = nn.Linear(config.d_model, 2, bias=False)

        self.init_weights()

    def get_input_embeddings(self):
        return self.trg_word_emb

    # def set_input_embeddings(self, new_embeddings):
    #     self.shared = new_embeddings
    #     self.encoder.set_input_embeddings(new_embeddings)
    #     self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_callable(T5_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_past_key_value_states=None,
        use_cache=True,
        lm_labels=None,
        # template_lm_labels=None,
        # action_count_labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
    ):
        r"""
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
                All labels set to ``-100`` are ignored (masked), the loss is only
                computed for labels in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.T5Config`) and inputs.
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_label` is provided):
            Classification loss (cross entropy).
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            If `past_key_value_states` is used only the last prediction_scores of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        decoder_past_key_value_states (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`, `optional`, returned when ``use_cache=True``):
            Contains pre-computed key and value hidden-states of the attention blocks.
            Can be used to speed up sequential decoding (see `decoder_past_key_value_states` input).
            Note that when using `decoder_past_key_value_states`, the model only outputs the last `prediction_score` of the sequence of shape :obj:`(batch_size, 1, config.vocab_size)`.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention.

    Examples::

        from transformers import T5Tokenizer, T5ForConditionalGeneration

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")  # Batch size 1
        outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        input_ids = tokenizer.encode("summarize: Hello, my dog is cute", return_tensors="pt")  # Batch size 1
        outputs = model.generate(input_ids)
        """

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask
            )

        hidden_states = encoder_outputs[0]

        if lm_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(lm_labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if decoder_past_key_value_states is not None:
            assert lm_labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        if self.random_mask_decoder_input and self.training:
            bsz, dec_seq_len = decoder_input_ids.size(0), decoder_input_ids.size(1)
            rand_num = torch.rand(bsz).cuda()
            random_mask = torch.cuda.FloatTensor(bsz, dec_seq_len).uniform_() > rand_num.unsqueeze(1)
            decoder_input_ids = decoder_input_ids.masked_fill_(random_mask, self.trg_vocab_size - 1)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_value_states=decoder_past_key_value_states,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
        )

        # insert decoder past at right place
        # to speed up decoding
        if use_cache is True:
            past = ((encoder_outputs, decoder_outputs[1]),)
            decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        if self.predict_action_count:
            # dec_self_attn_output = self.decoder.block[-1].self_attention_outputs
            action_count_logits = self.action_count_head(sequence_output)

        decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
        if lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            decoder_outputs = (loss,) + decoder_outputs
        # if action_count_labels is not None:
        #     action_count_loss_fct = CrossEntropyLoss(ignore_index=-1)
        #     self.action_count_loss = action_count_loss_fct(action_count_logits.view(-1, action_count_logits.size(-1)), action_count_labels.view(-1))

        return decoder_outputs + encoder_outputs

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, use_cache, step, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step
        if len(past) < 2:
            encoder_outputs, decoder_past_key_value_states = past, None
        else:
            encoder_outputs, decoder_past_key_value_states = past[0], past[1]

        return {
            "decoder_input_ids": input_ids,
            "decoder_past_key_value_states": decoder_past_key_value_states,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if len(past) < 2:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        decoder_past = past[1]
        past = (past[0],)
        reordered_decoder_past = ()
        for layer_past_states in decoder_past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return past + (reordered_decoder_past,)


class T5ForConditionalGenerationDualEmb(T5PreTrainedModel):
    """
    Adapted from Li et al. 2019, as used in experiments 3.3
    """
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.src_word_emb_func = nn.Embedding(config.vocab_size, config.d_model)
        self.src_word_emb_prim = nn.Embedding(config.vocab_size, config.d_model)

        self.trg_word_emb = nn.Embedding(config.trg_vocab_size, config.d_model)
        # self.trg_position_emb = nn.Embedding(80, config.d_model)

        encoder_config = copy.deepcopy(config)
        self.embedding_regularization = config.embedding_regularization
        self.layer_regularization = config.layer_regularization
        self.predict_action_count = config.predict_action_count

        self.encoder = T5StackWithDualEmb(
            encoder_config,
            self.src_word_emb_func,
            self.src_word_emb_prim,
            embedding_regularization=self.embedding_regularization,
            layer_regularization=self.layer_regularization,
        )

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = T5Stack(
            decoder_config,
            self.trg_word_emb,
            embedding_regularization=self.embedding_regularization,
            layer_regularization=self.layer_regularization,
            # embed_positions=self.trg_position_emb,
        )
        self.output_attn = MHAttention(decoder_config, has_relative_attention_bias=True)

        self.lm_head = nn.Linear(config.d_model, config.trg_vocab_size, bias=False)
        if self.predict_action_count:
            self.start_action_count_id = -1
            self.action_count_emb = None
            self.action_count_head = nn.Linear(config.d_model, 6, bias=False)

        self.init_weights()

    def get_input_embeddings(self):
        return self.trg_word_emb

    # def set_input_embeddings(self, new_embeddings):
    #     self.shared = new_embeddings
    #     self.encoder.set_input_embeddings(new_embeddings)
    #     self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_callable(T5_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_past_key_value_states=None,
        use_cache=True,
        lm_labels=None,
        action_count_labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
    ):
        r"""
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
                All labels set to ``-100`` are ignored (masked), the loss is only
                computed for labels in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.T5Config`) and inputs.
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_label` is provided):
            Classification loss (cross entropy).
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            If `past_key_value_states` is used only the last prediction_scores of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        decoder_past_key_value_states (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`, `optional`, returned when ``use_cache=True``):
            Contains pre-computed key and value hidden-states of the attention blocks.
            Can be used to speed up sequential decoding (see `decoder_past_key_value_states` input).
            Note that when using `decoder_past_key_value_states`, the model only outputs the last `prediction_score` of the sequence of shape :obj:`(batch_size, 1, config.vocab_size)`.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention.

    Examples::

        from transformers import T5Tokenizer, T5ForConditionalGeneration

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")  # Batch size 1
        outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        input_ids = tokenizer.encode("summarize: Hello, my dog is cute", return_tensors="pt")  # Batch size 1
        outputs = model.generate(input_ids)
        """

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask
            )

        hidden_states, hidden_states_prim, hidden_states_func = encoder_outputs[0], encoder_outputs[1], encoder_outputs[2]

        if lm_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(lm_labels)


        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if decoder_past_key_value_states is not None:
            assert lm_labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            position_ids=position_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_value_states=decoder_past_key_value_states,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
        )

        sequence_outputs = self.output_attn(
            input=decoder_outputs[0],
            mask=self.invert_attention_mask(attention_mask),
            k=hidden_states,
            v=hidden_states_prim
        )
        sequence_output = sequence_outputs[0]

        # insert decoder past at right place
        # to speed up decoding
        if use_cache is True:
            past = ((encoder_outputs[0], encoder_outputs[1], encoder_outputs[2], decoder_outputs[1]),)
            decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

        #sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        if self.predict_action_count:
            # dec_self_attn_output = self.decoder.block[-1].self_attention_outputs
            self.action_count_logits = self.action_count_head(sequence_output)

        decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
        if lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            decoder_outputs = (loss,) + decoder_outputs

        if action_count_labels is not None:
            action_count_loss_fct = CrossEntropyLoss(ignore_index=-1)
            self.action_count_loss = action_count_loss_fct(self.action_count_logits.view(-1, self.action_count_logits.size(-1)), action_count_labels.view(-1))

        if self.config.embedding_regularization:
            self.emb_reg_loss = self.encoder.func_emb_reg_loss + self.encoder.prim_emb_reg_loss #+ self.decoder.func_emb_reg_loss
        if self.config.layer_regularization:
            self.layer_reg_loss = self.encoder.layer_reg_losses + self.decoder.layer_reg_losses
        return decoder_outputs + encoder_outputs

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, use_cache, step, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step
        if len(past) < 4:
            encoder_outputs, decoder_past_key_value_states = past, None
            # position_ids = torch.tensor([0]).cuda()
        else:
            encoder_outputs, decoder_past_key_value_states = past[0:3], past[3]
        position_ids = torch.tensor([step-1]).cuda()

        return {
            "decoder_input_ids": input_ids,
            "decoder_past_key_value_states": decoder_past_key_value_states,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "position_ids": position_ids,
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if len(past) < 2:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        decoder_past = past[1]
        past = (past[0],)
        reordered_decoder_past = ()
        for layer_past_states in decoder_past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return past + (reordered_decoder_past,)


class T5ForConditionalGenerationDualEmbCntAction(T5PreTrainedModel):
    """
    Adapted from Li et al. 2019, as used in experiments 3.3
    """
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.src_word_emb_func = nn.Embedding(config.vocab_size, config.d_model)
        self.src_word_emb_prim = nn.Embedding(config.vocab_size, config.d_model)

        self.trg_word_emb = nn.Embedding(config.trg_vocab_size, config.d_model)
        self.action_count_attention_kv = config.action_count_attention_kv.split(',')
        assert len(self.action_count_attention_kv) == 2

        encoder_config = copy.deepcopy(config)
        self.embedding_regularization = config.embedding_regularization
        self.layer_regularization = config.layer_regularization
        self.predict_action_count = config.predict_action_count
        self.predict_action_group = config.predict_action_group

        self.encoder = T5StackWithDualEmb(
            encoder_config,
            self.src_word_emb_func,
            self.src_word_emb_prim,
            embedding_regularization=self.embedding_regularization,
            layer_regularization=self.layer_regularization,
        )

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True

        if self.predict_action_count:
            ### count_label_scheme = v1 ###
            self.action_count_emb = nn.Embedding(7, config.d_model)
            self.action_count_head = nn.Linear(config.d_model, 7, bias=False)
            self.start_action_count_id = 6
            self.count_output_attn = MHAttention(decoder_config, has_relative_attention_bias=True)

        if self.predict_action_group:
            self.action_group_emb = nn.Embedding(17, config.d_model)
            self.action_group_head = nn.Linear(config.d_model, 17, bias=False)
            self.start_action_group_id = 16
            self.group_output_attn = MHAttention(decoder_config, has_relative_attention_bias=True)

        # The decoder is a T5Stack
        self.decoder = T5Stack(
            decoder_config,
            self.trg_word_emb,
            embedding_regularization=self.embedding_regularization,
            layer_regularization=self.layer_regularization,
            embed_action_count=self.action_count_emb if self.predict_action_count else None,
            embed_action_group=self.action_group_emb if self.predict_action_group else None,
            # embed_positions=self.trg_position_emb,
        )
        self.output_attn = MHAttention(decoder_config, has_relative_attention_bias=True)

        self.lm_head = nn.Linear(config.d_model, config.trg_vocab_size, bias=False)

        self.init_weights()
        self.tie_action_count_weights()

    def get_input_embeddings(self):
        return self.trg_word_emb

    # def set_input_embeddings(self, new_embeddings):
    #     self.shared = new_embeddings
    #     self.encoder.set_input_embeddings(new_embeddings)
    #     self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_callable(T5_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        action_count_ids=None,
        action_group_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_past_key_value_states=None,
        use_cache=True,
        lm_labels=None,
        action_count_labels=None,
        action_group_labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
    ):
        r"""
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
                All labels set to ``-100`` are ignored (masked), the loss is only
                computed for labels in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.T5Config`) and inputs.
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_label` is provided):
            Classification loss (cross entropy).
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            If `past_key_value_states` is used only the last prediction_scores of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        decoder_past_key_value_states (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`, `optional`, returned when ``use_cache=True``):
            Contains pre-computed key and value hidden-states of the attention blocks.
            Can be used to speed up sequential decoding (see `decoder_past_key_value_states` input).
            Note that when using `decoder_past_key_value_states`, the model only outputs the last `prediction_score` of the sequence of shape :obj:`(batch_size, 1, config.vocab_size)`.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention.

    Examples::

        from transformers import T5Tokenizer, T5ForConditionalGeneration

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")  # Batch size 1
        outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        input_ids = tokenizer.encode("summarize: Hello, my dog is cute", return_tensors="pt")  # Batch size 1
        outputs = model.generate(input_ids)
        """

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask
            )

        hidden_states, hidden_states_prim, hidden_states_func = encoder_outputs[0], encoder_outputs[1], encoder_outputs[2]

        if lm_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(lm_labels)

        if action_count_labels is not None and action_count_ids is None:
            action_count_ids = self._shift_action_right(action_count_labels, self.start_action_count_id)

        if action_group_labels is not None and action_group_ids is None:
            action_group_ids = self._shift_action_right(action_group_labels, self.start_action_group_id)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if decoder_past_key_value_states is not None:
            assert lm_labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]
            if action_count_ids is not None:
                action_count_ids = action_count_ids[:, -1:]
            if action_group_ids is not None:
                action_group_ids = action_group_ids[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            position_ids=position_ids,
            action_count_ids=action_count_ids,
            action_group_ids=action_group_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_value_states=decoder_past_key_value_states,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
        )

        sequence_outputs = self.output_attn(
            input=decoder_outputs[0],
            mask=self.invert_attention_mask(attention_mask),
            k=hidden_states,
            v=hidden_states_prim
        )
        sequence_output = sequence_outputs[0]

        ### AuxSeq 1
        if self.action_count_attention_kv[0] == 'f':
            count_attn_key = hidden_states_func
        elif self.action_count_attention_kv[0] == 'c':
            count_attn_key = hidden_states
        elif self.action_count_attention_kv[0] == 'p':
            count_attn_key = hidden_states_prim
        else:
            raise NotImplementedError

        if self.action_count_attention_kv[1] == 'f':
            count_attn_value = hidden_states_func
        elif self.action_count_attention_kv[1] == 'c':
            count_attn_value = hidden_states
        elif self.action_count_attention_kv[1] == 'p':
            count_attn_value = hidden_states_prim
        else:
            raise NotImplementedError

        count_sequence_outputs = self.count_output_attn(
            #input=decoder_outputs[0],
            input=self.decoder.block[0].self_attention_outputs,
            # input=self.decoder.first_layer_output,
            mask=self.invert_attention_mask(attention_mask),
            k=count_attn_key,
            v=count_attn_value,
        )
        count_sequence_output = count_sequence_outputs[0]

        ### AuxSeq 2
        group_sequence_outputs = self.group_output_attn(
            input=decoder_outputs[0],
            mask=self.invert_attention_mask(attention_mask),
            k=hidden_states,
            v=hidden_states_prim
        )

        group_sequence_output = group_sequence_outputs[0]

        # insert decoder past at right place
        # to speed up decoding
        if use_cache is True:
            past = ((encoder_outputs[0], encoder_outputs[1], encoder_outputs[2], decoder_outputs[1]),)
            decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

        #sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        count_sequence_output = count_sequence_output * (self.model_dim ** -0.5)
        group_sequence_output = group_sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        if self.predict_action_count:
            #self.action_count_logits = self.action_count_head(decoder_outputs[0] * (self.model_dim ** -0.5))
            self.action_count_logits = self.action_count_head(count_sequence_output)
        if self.predict_action_group:
            # self.action_group_logits = self.action_group_head(decoder_outputs[0] * (self.model_dim ** -0.5))
            self.action_group_logits = self.action_group_head(group_sequence_output)

        decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
        if lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            decoder_outputs = (loss,) + decoder_outputs

        if action_count_labels is not None:
            # print(action_count_ids[0:5])
            # print(action_count_labels[0:5])
            action_count_loss_fct = CrossEntropyLoss(ignore_index=-1)
            self.action_count_loss = action_count_loss_fct(
                self.action_count_logits.view(-1, self.action_count_logits.size(-1)), action_count_labels.view(-1))

        if action_group_labels is not None:
            # print(action_group_labels[0])
            action_group_loss_fct = CrossEntropyLoss(ignore_index=-1)
            self.action_group_loss = action_group_loss_fct(
                self.action_group_logits.view(-1, self.action_group_logits.size(-1)), action_group_labels.view(-1))

        if self.config.embedding_regularization:
            self.emb_reg_loss = self.encoder.func_emb_reg_loss + self.encoder.prim_emb_reg_loss + self.decoder.func_emb_reg_loss
            # self.count_emb_reg_loss = self.decoder.action_group_emb_reg_loss + self.decoder.action_count_emb_reg_loss

        if self.config.layer_regularization:
            self.layer_reg_loss = self.encoder.layer_reg_losses + self.decoder.layer_reg_losses
        return decoder_outputs + encoder_outputs

    def prepare_inputs_for_generation(self, input_ids, action_count_ids, action_group_ids, past,
                                      attention_mask, use_cache, step, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"
        # first step
        if len(past) < 4:
            encoder_outputs, decoder_past_key_value_states = past, None
            # position_ids = torch.tensor([0]).cuda()
        else:
            encoder_outputs, decoder_past_key_value_states = past[0:3], past[3]
        position_ids = torch.tensor([step-1]).cuda()

        return {
            "decoder_input_ids": input_ids,
            "action_count_ids": action_count_ids,
            "action_group_ids": action_group_ids,
            "decoder_past_key_value_states": decoder_past_key_value_states,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "position_ids": position_ids,
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if len(past) < 2:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        decoder_past = past[1]
        past = (past[0],)
        reordered_decoder_past = ()
        for layer_past_states in decoder_past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return past + (reordered_decoder_past,)

    def _shift_action_right(self, input_ids, sos_id):
        #sos_id = self.start_action_count_id
        pad_token_id = sos_id

        assert (
            sos_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = sos_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in lm_labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -1, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `lm_labels` has only positive values and -100"

        return shifted_input_ids

    def tie_action_count_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.
        """
        if self.predict_action_count:
            output_embeddings = self.action_count_emb
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.action_count_head)
        if self.predict_action_group:
            group_embeddings = self.action_group_emb
            if group_embeddings is not None:
                self._tie_or_clone_weights(group_embeddings, self.action_group_head)
