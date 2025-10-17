import torch
import opt_einsum as oe
from cuequivariance_torch import triangle_multiplicative_update

class TriangleMultiplicationOutgoing(torch.nn.Module):
    def __init__(self, c=128):
        super().__init__()

        self.c = c

        self.layer_norm = torch.nn.LayerNorm(c)
        self.layer_norm_out = torch.nn.LayerNorm(c)

        self.proj_a = torch.nn.Linear(c, c)
        self.gate_a = torch.nn.Linear(c, c)
        self.proj_b = torch.nn.Linear(c, c)
        self.gate_b = torch.nn.Linear(c, c)
        self.gate = torch.nn.Linear(c, c)
        self.proj_o = torch.nn.Linear(c, c)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, z, pair_mask=None):
        """
        Algorithm 11: Triangular multiplicative update using "outgoing" edges
        (__init__ 변경 없이, Fused cuequivariance kernel을 사용하도록 forward만 수정)

        z: (B, i, j, c)
        return: (B, i, j, c)
        """
        # 1. 배치 차원 확인 및 임시 추가
        is_batched = z.dim() == 4
        if not is_batched:
            z = z.unsqueeze(0)
            if pair_mask is not None:
                pair_mask = pair_mask.unsqueeze(0)

        # 2. 마스크 준비
        if pair_mask is None:
            pair_mask = z.new_ones(z.shape[:-1])

        # 3. Fused Kernel에 전달할 파라미터 동적 결합
        # 분리된 proj_a와 proj_b의 가중치를 (2*c, c) 형태로 결합
        p_in_weight_fused = torch.cat([self.proj_a.weight, self.proj_b.weight], dim=0)
        p_in_bias_fused = torch.cat([self.proj_a.bias, self.proj_b.bias], dim=0)
        
        # 분리된 gate_a와 gate_b의 가중치를 (2*c, c) 형태로 결합
        g_in_weight_fused = torch.cat([self.gate_a.weight, self.gate_b.weight], dim=0)
        g_in_bias_fused = torch.cat([self.gate_a.bias, self.gate_b.bias], dim=0)
        
        # 4. cuequivariance 함수 호출
        # 모든 연산(norm, gate, matmul, out_norm, out_proj)이 이 함수 안에서 처리됨
        output = triangle_multiplicative_update(
            x=z,
            direction="outgoing",
            mask=pair_mask,
            # 입력 정규화 파라미터
            norm_in_weight=self.layer_norm.weight,
            norm_in_bias=self.layer_norm.bias,
            # 동적으로 결합한 입력 프로젝션/게이트 파라미터
            p_in_weight=p_in_weight_fused,
            # p_in_bias=p_in_bias_fused,
            g_in_weight=g_in_weight_fused,
            # g_in_bias=g_in_bias_fused,
            # 출력 정규화 파라미터
            norm_out_weight=self.layer_norm_out.weight,
            norm_out_bias=self.layer_norm_out.bias,
            # 출력 프로젝션/게이트 파라미터
            p_out_weight=self.proj_o.weight,
            # p_out_bias=self.proj_o.bias,
            g_out_weight=self.gate.weight,
            # g_out_bias=self.gate.bias
        )

        # 5. (필요시) 임시 배치 차원 제거
        if not is_batched:
            output = output.squeeze(0)

        return output


class TriangleMultiplicationIncoming(torch.nn.Module):
    def __init__(self, c=128):
        super().__init__()

        self.c = c

        self.layer_norm = torch.nn.LayerNorm(c)
        self.layer_norm_out = torch.nn.LayerNorm(c)

        self.proj_a = torch.nn.Linear(c, c)
        self.gate_a = torch.nn.Linear(c, c)
        self.proj_b = torch.nn.Linear(c, c)
        self.gate_b = torch.nn.Linear(c, c)
        self.gate = torch.nn.Linear(c, c)
        self.proj_o = torch.nn.Linear(c, c)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, z, pair_mask=None):
        """
        Algorithm 12: Triangular multiplicative update using "incoming" edges
        (__init__ 변경 없이, Fused cuequivariance kernel을 사용하도록 forward만 수정)

        z: (B, i, j, c)
        return: (B, i, j, c)
        """
        # 1. 배치 차원 확인 및 임시 추가
        is_batched = z.dim() == 4
        if not is_batched:
            z = z.unsqueeze(0)
            if pair_mask is not None:
                pair_mask = pair_mask.unsqueeze(0)

        # 2. 마스크 준비
        if pair_mask is None:
            pair_mask = z.new_ones(z.shape[:-1])

        # 3. Fused Kernel에 전달할 파라미터 동적 결합
        # 분리된 proj_a와 proj_b의 가중치를 (2*c, c) 형태로 결합
        p_in_weight_fused = torch.cat([self.proj_a.weight, self.proj_b.weight], dim=0)
        p_in_bias_fused = torch.cat([self.proj_a.bias, self.proj_b.bias], dim=0)
        
        # 분리된 gate_a와 gate_b의 가중치를 (2*c, c) 형태로 결합
        g_in_weight_fused = torch.cat([self.gate_a.weight, self.gate_b.weight], dim=0)
        g_in_bias_fused = torch.cat([self.gate_a.bias, self.gate_b.bias], dim=0)
        
        # 4. cuequivariance 함수 호출
        # 모든 연산(norm, gate, matmul, out_norm, out_proj)이 이 함수 안에서 처리됨
        output = triangle_multiplicative_update(
            x=z,
            direction="incoming",
            mask=pair_mask,
            # 입력 정규화 파라미터
            norm_in_weight=self.layer_norm.weight,
            norm_in_bias=self.layer_norm.bias,
            # 동적으로 결합한 입력 프로젝션/게이트 파라미터
            p_in_weight=p_in_weight_fused,
            # p_in_bias=p_in_bias_fused,
            g_in_weight=g_in_weight_fused,
            # g_in_bias=g_in_bias_fused,
            # 출력 정규화 파라미터
            norm_out_weight=self.layer_norm_out.weight,
            norm_out_bias=self.layer_norm_out.bias,
            # 출력 프로젝션/게이트 파라미터
            p_out_weight=self.proj_o.weight,
            # p_out_bias=self.proj_o.bias,
            g_out_weight=self.gate.weight,
            # g_out_bias=self.gate.bias
        )

        # 5. (필요시) 임시 배치 차원 제거
        if not is_batched:
            output = output.squeeze(0)

        return output