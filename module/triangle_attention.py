import torch
import opt_einsum as oe
from cuequivariance_torch import triangle_attention

class TriangleAttentionStartingNode(torch.nn.Module):
    def __init__(self, c=32, c_h=16, n_head=4):
        super().__init__()
        self.c = c
        self.c_h = c_h
        self.n_head = n_head

        self.layer_norm = torch.nn.LayerNorm(c)

        self.query = torch.nn.Linear(c, c_h * n_head, bias=False)
        self.key = torch.nn.Linear(c, c_h * n_head, bias=False)
        self.value = torch.nn.Linear(c, c_h * n_head, bias=False)

        self.bias = torch.nn.Linear(c, n_head, bias=False)
        self.gate = torch.nn.Linear(c, c_h * n_head)
        self.output = torch.nn.Linear(c_h * n_head, c)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, z, pair_mask=None):
        """
        Algorithm 13: Triangular gated self-attention around starting node

        z: (B, i, j, c)

        return: (B, i, j, c)
        """

        # 1. 배치 차원 확인 (이전과 동일)
        is_batched = z.dim() == 4
        if not is_batched:
            z = z.unsqueeze(0)
            if pair_mask is not None:
                pair_mask = pair_mask.unsqueeze(0)

        # 2. 초기 프로젝션 (이전과 동일)
        z_norm = self.layer_norm(z)
        q = self.query(z_norm)
        k = self.key(z_norm)
        v = self.value(z_norm)
        b = self.bias(z_norm)
        g = self.sigmoid(self.gate(z_norm))

        # 3. cuequivariance를 위한 텐서 형태 변환
        B, i_dim, j_dim, _ = z.shape
        h, c_h = self.n_head, self.c_h

        # Head 차원 분리
        q, k, v, g = [x.view(B, i_dim, j_dim, h, c_h) for x in (q, k, v, g)]
        
        # 'i'를 배치, 'j'를 시퀀스로 취급하기 위해 차원 변경
        # (B, i, j, h, c) -> (B, i, h, j, c)
        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        # 4. Bias 및 Mask 준비 (논문 원리에 기반하여 재검증)
        
        # Bias: (B, i, j, h) -> (B, i, h, j)
        # 어텐션 행렬 (B, i, h, j, j)에 더해지기 위해 key 시퀀스(마지막 j)에 대한 1D bias로 만듦
        # (B, i, h, j) -> (B, i, h, 1, j)
        b = b.permute(0, 3, 1, 2)
        
        attn_mask = None
        if pair_mask is not None:
            # pair_mask는 각 'i'에 대한 1D 마스크 (B, i, j)
            # 이를 브로드캐스팅하여 2D 어텐션 마스크 (B, i, j, j) 생성
            # key(v)와 query(q)에 모두 적용
            mask_2d = pair_mask.unsqueeze(-1) * pair_mask.unsqueeze(-2)
            attn_mask = mask_2d.bool()

        # 5. cuequivariance 함수로 어텐션 수행
        # 출력 o의 형태: (B, i, h, j, c_h)
        o = triangle_attention(
            q, k, v,
            bias=b,
            mask=attn_mask,
        )

        # 6. 출력 텐서 형태 복원 및 후처리
        # (B, i, h, j, c_h) -> (B, i, j, h, c_h) : 원래 순서로 복원
        o = o.permute(0, 1, 3, 2, 4)

        # 원래 형태의 텐서로 reshape
        o = o.reshape(B, i_dim, j_dim, h * c_h)
        g = g.reshape(B, i_dim, j_dim, h * c_h)

        # 게이팅 및 최종 프로젝션
        o = g * o
        o = self.output(o)

        # 7. (필요시) 임시 배치 차원 제거 (이전과 동일)
        if not is_batched:
            o = o.squeeze(0)
            
        return o
    

class TriangleAttentionEndingNode(torch.nn.Module):
    def __init__(self, c=32, c_h=16, n_head=4):
        super().__init__()
        self.c = c
        self.c_h = c_h
        self.n_head = n_head

        self.layer_norm = torch.nn.LayerNorm(c)

        self.query = torch.nn.Linear(c, c_h * n_head, bias=False)
        self.key = torch.nn.Linear(c, c_h * n_head, bias=False)
        self.value = torch.nn.Linear(c, c_h * n_head, bias=False)

        self.bias = torch.nn.Linear(c, n_head, bias=False)
        self.gate = torch.nn.Linear(c, c_h * n_head)
        self.output = torch.nn.Linear(c_h * n_head, c)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, z, pair_mask=None):
        """
        Algorithm 14: Triangular gated self-attention around ending node

        z: (B, i, j, c)

        return: (B, i, j, c)
        """

        z = z.transpose(-2, -3) # Ending node, flips i and j
        # 1. 배치 차원 확인 (이전과 동일)
        is_batched = z.dim() == 4
        if not is_batched:
            z = z.unsqueeze(0)
            if pair_mask is not None:
                pair_mask = pair_mask.unsqueeze(0)

        # 2. 초기 프로젝션 (이전과 동일)
        z_norm = self.layer_norm(z)
        q = self.query(z_norm)
        k = self.key(z_norm)
        v = self.value(z_norm)
        b = self.bias(z_norm)
        g = self.sigmoid(self.gate(z_norm))

        # 3. cuequivariance를 위한 텐서 형태 변환
        B, i_dim, j_dim, _ = z.shape
        h, c_h = self.n_head, self.c_h

        # Head 차원 분리
        q, k, v, g = [x.view(B, i_dim, j_dim, h, c_h) for x in (q, k, v, g)]
        
        # 'i'를 배치, 'j'를 시퀀스로 취급하기 위해 차원 변경
        # (B, i, j, h, c) -> (B, i, h, j, c)
        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        # 4. Bias 및 Mask 준비 (논문 원리에 기반하여 재검증)
        
        # Bias: (B, i, j, h) -> (B, i, h, j)
        # 어텐션 행렬 (B, i, h, j, j)에 더해지기 위해 key 시퀀스(마지막 j)에 대한 1D bias로 만듦
        # (B, i, h, j) -> (B, i, h, 1, j)
        b = b.permute(0, 3, 1, 2)
        
        attn_mask = None
        if pair_mask is not None:
            # pair_mask는 각 'i'에 대한 1D 마스크 (B, i, j)
            # 이를 브로드캐스팅하여 2D 어텐션 마스크 (B, i, j, j) 생성
            # key(v)와 query(q)에 모두 적용
            mask_2d = pair_mask.unsqueeze(-1) * pair_mask.unsqueeze(-2)
            attn_mask = mask_2d.bool()

        # 5. cuequivariance 함수로 어텐션 수행
        # 출력 o의 형태: (B, i, h, j, c_h)
        o = triangle_attention(
            q, k, v,
            bias=b,
            mask=attn_mask,
        )

        # 6. 출력 텐서 형태 복원 및 후처리
        # (B, i, h, j, c_h) -> (B, i, j, h, c_h) : 원래 순서로 복원
        o = o.permute(0, 1, 3, 2, 4)

        # 원래 형태의 텐서로 reshape
        o = o.reshape(B, i_dim, j_dim, h * c_h)
        g = g.reshape(B, i_dim, j_dim, h * c_h)

        # 게이팅 및 최종 프로젝션
        o = g * o
        o = self.output(o).transpose(-2, -3)

        # 7. (필요시) 임시 배치 차원 제거 (이전과 동일)
        if not is_batched:
            o = o.squeeze(0)
            
        
        return o  # (B, i, j, c_z)