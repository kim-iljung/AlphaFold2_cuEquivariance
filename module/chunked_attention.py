import torch
from typing import List
from torch.utils.checkpoint import checkpoint

def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, biases: List[torch.Tensor]) -> torch.Tensor:
    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a += b

    a = softmax_no_cast(a, dim=-1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a

def _attention_chunked_trainable(
    query, key, value, biases, chunk_size, chunk_dim, checkpoint_flag, 
):
    if checkpoint_flag and len(biases) > 2:
        raise ValueError(
            "Checkpointed version permits only permits two bias terms"
        )

    def _checkpointable_attention(q, k, v, b1, b2):
        bs = [b for b in [b1, b2] if b is not None]
        a = _attention(q, k, v, bs)
        return a

    o_chunks = []
    count = query.shape[chunk_dim]
    for start in range(0, count, chunk_size):
        end = start + chunk_size
        idx = [slice(None)] * len(query.shape)
        idx[chunk_dim] = slice(start, end)
        idx_tup = tuple(idx)
        q_chunk = query[idx_tup]
        k_chunk = key[idx_tup]
        v_chunk = value[idx_tup]

        def _slice_bias(b):
            idx[chunk_dim] = (
                slice(start, end) if b.shape[chunk_dim] != 1 else slice(None)
            )
            return b[tuple(idx)]

        if checkpoint_flag:
            bias_1_chunk, bias_2_chunk = [
                _slice_bias(b) if b is not None else None
                for b in (biases + [None, None])[:2]
            ]

            o_chunk = checkpoint(_checkpointable_attention,
                q_chunk, k_chunk, v_chunk, bias_1_chunk, bias_2_chunk
            )
        else:
            bias_chunks = [
                _slice_bias(b) for b in biases
            ]

            o_chunk = _attention(q_chunk, k_chunk, v_chunk, bias_chunks)
            
        o_chunk = o_chunk.transpose(-2, -3)
        o_chunks.append(o_chunk)

    o = torch.cat(o_chunks, dim=chunk_dim)
    return o

def softmax_no_cast(t, dim=-1):
    d = t.dtype
    if d is torch.bfloat16:
        with torch.cuda.amp.autocast(enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)

    return s