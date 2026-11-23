"""Incremental decoding for learned and relative positions, including coverage."""
import math
import torch
from torch.nn import functional as F
from model.relative_position import alibi_slopes


def project(attention, values, part):
    width=attention.embed_dim
    bias=None if attention.in_proj_bias is None else attention.in_proj_bias[part*width:(part+1)*width]
    result=F.linear(values,attention.in_proj_weight[part*width:(part+1)*width],bias)
    return result.reshape(result.shape[0],result.shape[1],attention.num_heads,width//attention.num_heads).transpose(1,2)


class DecoderState:
    def __init__(self, decoder, memory, memory_mask):
        if decoder.training:
            raise ValueError('Cached study decoding requires an evaluation-mode decoder')
        self.decoder,self.memory,self.mask=decoder,memory,memory_mask
        self.layers=[dict(k=None,v=None,mk=project(l.multihead_attn,memory,1),
                         mv=project(l.multihead_attn,memory,2)) for l in decoder.transformer.layers]
        self.coverage=memory.new_zeros(memory.shape[0],decoder.transformer.layers[-1].self_attn.num_heads,1,memory.shape[1])
        self.previous=None
        self.length=0

    def repeat_beams(self, width):
        """Expand projected encoder memory before any prefix has been decoded."""
        if self.length:
            raise ValueError('Beam expansion requires an unused decoder state')
        self.memory = self.memory.repeat_interleave(width, 0)
        if self.mask is not None:
            self.mask = self.mask.repeat_interleave(width, 0)
        self.coverage = self.coverage.repeat_interleave(width, 0)
        for layer in self.layers:
            layer['mk'] = layer['mk'].repeat_interleave(width, 0)
            layer['mv'] = layer['mv'].repeat_interleave(width, 0)

    def reorder_beams(self, parents):
        """Move the prefix state among beams sharing the same encoder memory."""
        for layer in self.layers:
            layer['k'] = layer['k'].index_select(0, parents)
            layer['v'] = layer['v'].index_select(0, parents)
        if self.decoder.coverage:
            self.coverage = self.coverage.index_select(0, parents)
            self.previous = self.previous.index_select(0, parents)

    def attend(self, attention, query, keys, values, bias=None, *, weights_required=True):
        q=project(attention,query,0)
        if weights_required:
            score=q @ keys.transpose(-1,-2)/math.sqrt(q.shape[-1])
            if bias is not None: score=score+bias
            weights=score.softmax(-1)
            attended = weights @ values
        else:
            weights = None
            attended = F.scaled_dot_product_attention(q, keys, values, attn_mask=bias)
        attended = attended.transpose(1,2).reshape(query.shape[0],1,-1)
        return attention.out_proj(attended),weights

    def step(self, token, *, fast=False):
        decoder=self.decoder
        if token.shape[1]!=1: raise ValueError('One cached token per step')
        values=decoder.token_embedding(token)
        if decoder.position_embedding is not None:
            position = torch.full_like(token, self.length)
            values = values + decoder.position_embedding(position)
        if self.previous is not None and decoder.coverage:
            self.coverage=self.coverage+self.previous*decoder.visible_tokens[token][:,None,:,None]
        final=None
        for index,(layer,state) in enumerate(zip(decoder.transformer.layers,self.layers)):
            q=layer.norm1(values)
            key,value=project(layer.self_attn,q,1),project(layer.self_attn,q,2)
            state['k']=key if state['k'] is None else torch.cat((state['k'],key),2)
            state['v']=value if state['v'] is None else torch.cat((state['v'],value),2)
            bias = None
            if decoder.position_embedding is None:
                distance=self.length-torch.arange(self.length+1,device=values.device)
                bias=-alibi_slopes(layer.self_attn.num_heads,values.device)[None,:,None,None]*distance[None,None,None]
            attended,_=self.attend(layer.self_attn,q,state['k'],state['v'],bias,weights_required=not fast)
            values=values+attended
            q=layer.norm2(values)
            mask_bias=None if self.mask is None else values.new_zeros(self.mask.shape).masked_fill(
                self.mask,float('-inf'))[:,None,None]
            last = index == len(self.layers)-1
            attended,weights=self.attend(layer.multihead_attn,q,state['mk'],state['mv'],mask_bias,
                                         weights_required=not fast or (last and decoder.coverage))
            if last and weights is not None:
                if self.mask is not None: weights=weights.masked_fill(self.mask[:,None,None],0)
                weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-6)
                self.previous=weights
                if decoder.coverage:
                    coverage_bias=-F.softplus(decoder.coverage_strength)*self.coverage
                    attended,weights=self.attend(layer.multihead_attn,q,state['mk'],state['mv'],
                                                 coverage_bias if mask_bias is None else coverage_bias+mask_bias)
                final=weights.mean(1)
            values=values+attended
            q=layer.norm3(values)
            values=values+layer.linear2(layer.activation(layer.linear1(q)))
        logits=decoder.output(decoder.transformer.norm(values))
        if decoder.coverage:
            valid=torch.ones_like(self.memory[...,0],dtype=torch.bool) if self.mask is None else ~self.mask
            uncovered=(torch.exp(-4*self.coverage.mean(1))*valid[:,None]).sum(-1)/valid.sum(-1)[:,None].clamp_min(1)
            eos=F.one_hot(torch.tensor(decoder.eos_id,device=values.device),logits.shape[-1]).to(logits.dtype)
            logits=logits-F.softplus(decoder.eos_coverage_strength)*uncovered[...,None]*eos
        self.length+=1
        return logits,final


@torch.inference_mode()
def generate_batch(model,features,mask,tokenizer,max_tokens=None):
    if model.writer_context is not None:raise ValueError('Cached batch decoding does not support writer context')
    model.eval()
    memory,memory_mask=model.encode(features,mask)
    state=DecoderState(model.decoder,memory,memory_mask)
    token=torch.full((len(features),1),tokenizer.bos_id,device=features.device,dtype=torch.long)
    generated=[];finished=torch.zeros(len(features),device=features.device,dtype=torch.bool)
    for _ in range((max_tokens or model.config.max_tokens)-1):
        logits,_=state.step(token)
        token=logits[:,-1].argmax(-1)[:,None]
        token=torch.where(finished[:,None],torch.full_like(token,tokenizer.pad_id),token)
        generated.append(token);finished|=token[:,0].eq(tokenizer.eos_id)
        if bool(finished.all()):break
    return dict(token_ids=torch.cat(generated,1) if generated else token[:,:0],truncated=~finished)


@torch.inference_mode()
def generate(model,features,mask,tokenizer,max_tokens=None):
    model.eval()
    memory,memory_mask=model.encode(features,mask)
    state=DecoderState(model.decoder,memory,memory_mask)
    token=torch.tensor([[tokenizer.bos_id]],device=features.device)
    ids=[]
    for _ in range((max_tokens or model.config.max_tokens)-1):
        logits,_=state.step(token)
        token=logits[:,-1].argmax(-1)[:,None]
        value=int(token[0,0]);ids.append(value)
        if value==tokenizer.eos_id: return dict(latex=tokenizer.decode(ids),truncated=False)
    return dict(latex=tokenizer.decode(ids),truncated=True)
