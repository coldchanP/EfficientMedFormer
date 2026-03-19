import torch
from torch import nn
import torch.nn.functional as F


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# AnisotropicConvBlock - 1×7, 7×1 기반 경량 블럭
class AnisotropicConvBlock(nn.Module):
    """적응적 커널 크기 기반 경량 블럭 + Residual Connection"""

    def __init__(self, in_channels, out_channels=None, downsample=False, kernel_size=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.downsample = downsample

        # H 방향 처리 (1×kernel_size 커널)
        self.h_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size),
                      stride=2 if downsample else 1,
                      padding=(0, kernel_size // 2), groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        # W 방향 처리 (kernel_size×1 커널)
        self.w_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1),
                      stride=2 if downsample else 1,
                      padding=(kernel_size // 2, 0), groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        # 채널 조정 (필요시)
        if self.in_channels != self.out_channels:
            self.channel_adjust = nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.GELU()
            )
        else:
            self.channel_adjust = nn.Identity()

        # Residual connection을 위한 입력 변환 (채널 수나 해상도가 변경될 때)
        if (self.in_channels != self.out_channels) or self.downsample:
            self.residual_transform = nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels,
                          kernel_size=1, stride=2 if downsample else 1, bias=False),
                nn.BatchNorm2d(self.out_channels)
            )
        else:
            self.residual_transform = nn.Identity()

        # 융합 레이어
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

    def forward(self, x):
        # Residual connection을 위한 입력 저장
        identity = x

        # H, W 방향 병렬 처리
        h_out = self.h_conv(x)  # 1×7 처리
        w_out = self.w_conv(x)  # 7×1 처리

        # 융합 (element-wise add)
        fused = h_out + w_out
        # fused = self.fusion(w_out)

        # 채널 조정
        output = self.channel_adjust(fused)

        # Residual connection
        identity_transformed = self.residual_transform(identity)
        output = output + identity_transformed

        return output


# Hierarchical Encoder - AnisotropicConvBlock 기반 특징 추출
class HierarchicalEncoder(nn.Module):
    """AnisotropicConvBlock 기반 계층적 특징 추출기"""

    def __init__(self, image_size=256, channels=3):
        super().__init__()

        # Stem: 256×256×3 → 128×128×24
        self.stem = nn.Sequential(
            nn.Conv2d(channels, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU()
        )

        # AnisotropicConvBlock 1: 128×128×24 → 128×128×32 (채널 확장, 큰 해상도용 7×1, 1×7)
        self.aniso_block1 = AnisotropicConvBlock(in_channels=24, out_channels=32, downsample=False, kernel_size=7)

        # AnisotropicConvBlock 2: 128×128×32 → 64×64×32 (다운샘플링, 큰 해상도용 7×1, 1×7)
        self.aniso_block2 = AnisotropicConvBlock(in_channels=32, out_channels=32, downsample=True, kernel_size=7)

        # AnisotropicConvBlock 3: 64×64×32 → 64×64×48 (채널 확장, 중간 해상도용 7×1, 1×7)
        self.aniso_block3 = AnisotropicConvBlock(in_channels=32, out_channels=48, downsample=False, kernel_size=7)

        # AnisotropicConvBlock 4: 64×64×48 → 32×32×48 (다운샘플링, 중간 해상도용 7×1, 1×7)
        self.aniso_block4 = AnisotropicConvBlock(in_channels=48, out_channels=48, downsample=True, kernel_size=7)

    def forward(self, x):
        # x: [B, 3, 256, 256]

        x = self.stem(x)  # [B, 24, 128, 128]
        x = self.aniso_block1(x)  # [B, 32, 128, 128] (채널 확장)
        x = self.aniso_block2(x)  # [B, 32, 64, 64] (다운샘플링)
        x = self.aniso_block3(x)  # [B, 48, 64, 64] (채널 확장)
        x = self.aniso_block4(x)  # [B, 48, 32, 32] (다운샘플링)

        return x


# Position Perceive - Conv 기반 위치 임베딩 (경량화)
class PositionPerceive(nn.Module):
    """경량 Conv 기반 위치 임베딩 - 적응적 DWConv"""

    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.channels = channels

        # 적응적 Depthwise Conv로 위치 임베딩 (해상도별 최적화)
        self.position_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2,
                      groups=channels, bias=False),  # Depthwise Conv
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, x):
        # 5x5 DWConv로 위치 임베딩 생성
        position_embedding = self.position_conv(x)

        # Add position embedding to input (residual connection)
        return x + position_embedding


class DualAttentionHead(nn.Module):
    """통합된 OH + OW Attention Head - H/W용 별도 context 생성"""

    def __init__(self, channels, kernel_size=7):
        super().__init__()

        # Vertical Processor (H component): 세로 방향(H)의 특징을 추출합니다. (kernel_size, 1) 커널 사용
        self.h_processor = nn.Sequential(
            nn.Conv2d(channels, channels,
                      kernel_size=(kernel_size, 1),
                      padding=(kernel_size // 2, 0),
                      bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        # Horizontal Processor (W component): 가로 방향(W)의 특징을 추출합니다. (1, kernel_size) 커널 사용
        self.w_processor = nn.Sequential(
            nn.Conv2d(channels, channels,
                      kernel_size=(1, kernel_size),
                      padding=(0, kernel_size // 2),
                      bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        # Global Average Pooling Layer: 피처맵을 채널별 스코어로 변환합니다.
        self.pool = nn.AdaptiveAvgPool2d(1)

        # # OH Fusion
        # self.oh_fusion = nn.Sequential(
        #     nn.Conv2d(channels, channels, 1, bias=False),
        #     nn.BatchNorm2d(channels),
        #     nn.GELU()
        # )

        # # OW Fusion
        # self.ow_fusion = nn.Sequential(
        #     nn.Conv2d(channels, channels, 1, bias=False),
        #     nn.BatchNorm2d(channels),
        #     nn.GELU()
        # )

    def forward(self, x):
        # 1. 각 방향에 특화된 특징맵(h_features, w_features)을 추출합니다.
        # 이 과정에서 학습 가능한 파라미터가 사용됩니다.
        h_features = self.h_processor(x)  # [B, C, H, W]
        w_features = self.w_processor(x)  # [B, C, H, W]

        # 2. 각 특징맵에 Global Average Pooling을 적용하여 컨텍스트 벡터를 생성합니다.
        # h_context: 세로 방향 특징에 각 채널이 얼마나 반응했는지를 나타내는 벡터. [B, C, 1, 1]
        # w_context: 가로 방향 특징에 각 채널이 얼마나 반응했는지를 나타내는 벡터. [B, C, 1, 1]
        h_context = self.pool(h_features)
        w_context = self.pool(w_features)

        # 3. 컨텍스트 벡터를 시그모이드 함수를 통과시켜 [0, 1] 범위의 게이트로 만듭니다.
        h_context_gate = torch.sigmoid(h_context)
        w_context_gate = torch.sigmoid(w_context)

        # 4. 원래의 특징맵에 해당 방향의 게이트를 곱하여 채널별 중요도를 조절합니다.
        context_h = h_features * h_context_gate
        context_w = w_features * w_context_gate

        # 5. 최종 융합을 통해 각 방향의 출력을 생성합니다.
        # oh_output = self.oh_fusion(context_h)
        # ow_output = self.ow_fusion(context_w)

        return context_h, context_w


class GCAA_Module(nn.Module):
    """통합된 듀얼 어텐션 모듈: DualAttentionHead 기반"""

    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.channels = channels
        bottleneck_channels = channels

        # 채널 축소 레이어
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU()
        )

        # 통합된 DualAttentionHead
        self.dual_attention = DualAttentionHead(bottleneck_channels, kernel_size)

        # Concat 후 각각 1x1 CBG 처리 후 최종 융합
        # bottleneck_channels * 2 입력에서 각 half에 대해 별도 처리
        self.oh_conv = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU()
        )
        
        self.ow_conv = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU()
        )
        
        # 최종 융합 (bottleneck_channels * 2 -> bottleneck_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(bottleneck_channels * 2, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU()
        )

        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU()
        )



    def forward(self, x):
        x_reduced = self.channel_reduce(x)

        # 통합된 듀얼 어텐션에서 OH, OW 출력을 동시에 받기
        oh_output, ow_output = self.dual_attention(x_reduced)

        # 먼저 concat
        combined = torch.cat([oh_output, ow_output], dim=1)  # [B, bottleneck_channels*2, H, W]
        
        # Concat된 텐서를 반으로 나누어 각각 처리
        oh_part = combined[:, :oh_output.shape[1], :, :]  # 첫 번째 half
        ow_part = combined[:, oh_output.shape[1]:, :, :]  # 두 번째 half
        
        # 각각에 1x1 CBG 적용
        oh_processed = self.oh_conv(oh_part)
        ow_processed = self.ow_conv(ow_part)
        
        # 처리된 결과를 다시 concat하고 최종 융합
        final_combined = torch.cat([oh_processed, ow_processed], dim=1)
        fused_output = self.fusion_conv(final_combined)

        # Residual connection
        fused_output = fused_output + x_reduced

        final_output = self.pointwise_conv(fused_output)

        return final_output


class AxialAttentionBlock(nn.Module):
    """Multi-head Axial Attention block that preserves channel size (in_channels -> in_channels)."""
    
    def __init__(self, in_channels, heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        
        # Ensure inner dimension is divisible by number of heads
        head_dim = max(1, in_channels // heads)
        inner_dim = head_dim * heads
        self.head_dim = head_dim
        self.inner_dim = inner_dim
        self.scale = head_dim ** -0.5
        
        # Shared QKV projections for axial attention along one axis (1D over channel dimension)
        self.to_qkv = nn.Conv1d(in_channels, inner_dim * 3, kernel_size=1, bias=False)
        self.proj_out = nn.Conv1d(inner_dim, in_channels, kernel_size=1, bias=False)
    
    def _axial_attention(self, x, axis: str):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        
        if axis == "h":
            # Attention along height axis
            x_perm = x.permute(0, 3, 1, 2).contiguous()  # [B, W, C, H]
            x_seq = x_perm.view(b * w, c, h)  # [B*W, C, H]
            length = h
        else:
            # Attention along width axis
            x_perm = x.permute(0, 2, 1, 3).contiguous()  # [B, H, C, W]
            x_seq = x_perm.view(b * h, c, w)  # [B*H, C, W]
            length = w
        
        # Compute Q, K, V
        qkv = self.to_qkv(x_seq)  # [B*axis, 3*inner_dim, L]
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        heads = self.heads
        head_dim = self.head_dim
        inner_dim = self.inner_dim
        
        q = q.view(-1, heads, head_dim, length).permute(0, 1, 3, 2)  # [B*axis, heads, L, head_dim]
        k = k.view(-1, heads, head_dim, length).permute(0, 1, 3, 2)  # [B*axis, heads, L, head_dim]
        v = v.view(-1, heads, head_dim, length).permute(0, 1, 3, 2)  # [B*axis, heads, L, head_dim]
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B*axis, heads, L, L]
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # [B*axis, heads, L, head_dim]
        
        # Merge heads
        out = out.permute(0, 1, 3, 2).contiguous().view(-1, inner_dim, length)  # [B*axis, inner_dim, L]
        out = self.proj_out(out)  # [B*axis, C, L]
        
        if axis == "h":
            out = out.view(b, w, c, h).permute(0, 2, 3, 1).contiguous()  # [B, C, H, W]
        else:
            out = out.view(b, h, c, w).permute(0, 2, 1, 3).contiguous()  # [B, C, H, W]
        
        return out
    
    def forward(self, x):
        # 1) Height-axis multi-head attention
        h_out = self._axial_attention(x, axis="h")        # [B, C, H, W]
        
        # 2) Width-axis multi-head attention, using H-axis output as input
        w_out = self._axial_attention(h_out, axis="w")    # [B, C, H, W]
        
        # 3) Residual add with original x (channel 수는 그대로 유지)
        y = x + w_out
        return y
    
    
# GCAA_FormerBlock - Axial Attention 기반 블럭
class GCAA_FormerBlock(nn.Module):
    """Concat 기반 채널 확장 + Position Perceive + AxialAttentionBlock (블록 타입만 변경)."""
    
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
    
        # Position Perceive - DWConv 기반 위치 임베딩 (입력 채널 그대로)
        self.position_perceive = PositionPerceive(
            channels=in_channels,
            kernel_size=kernel_size
        )
    
        # AxialAttention 전에 채널을 2배로 부풀리는 1x1 conv (C -> 2C)
        self.pre_axial_expand = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU()
        )

        # 확장된 채널(2C)에 대해 AxialAttentionBlock 적용 (2C -> 2C)
        self.attention = AxialAttentionBlock(
            in_channels=in_channels * 2,
            heads=8
        )

        # AxialAttention 출력(2C)을 다시 원래 채널 수(C)로 축소
        self.post_axial_reduce = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
    
        # Concat 후 채널 조정: [in_channels + bottleneck_channels] → out_channels
        bottleneck_channels = in_channels  # AxialAttentionBlock 출력 채널과 동일
        concat_channels = in_channels + bottleneck_channels  # 원본 + Attention 결과
        self.channel_adjust = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
        # 트랜스포머 블럭 전체 residual을 위한 채널 맞춤 (필요시)
        if in_channels != out_channels:
            self.input_projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.input_projection = nn.Identity()
    
    def forward(self, x):
        # 트랜스포머 블럭 전체 residual을 위한 입력 저장
        block_input = x  # [B, in_channels, H, W]
    
        # 1. Position Perceive - DWConv 기반 위치 임베딩
        x = self.position_perceive(x)  # [B, in_channels, H, W]
    
        # 2. AxialAttentionBlock 앞에서 채널을 2배로 부풀린 뒤 AxialAttention 적용
        identity = x                                         # [B, C, H, W]
        x_expanded = self.pre_axial_expand(x)                # [B, 2C, H, W]
        x_att_expanded = self.attention(x_expanded)          # [B, 2C, H, W]
        x_attention = self.post_axial_reduce(x_att_expanded) # [B, C, H, W]
    
        # 3. Concat + 채널 조정 (원본 + Attention 결과)
        x_concat = torch.cat([identity, x_attention], dim=1)  # [B, in_channels + bottleneck_channels, H, W]
        x_final = self.channel_adjust(x_concat)               # [B, out_channels, H, W]
    
        # 4. 트랜스포머 블럭 전체 residual connection
        block_input_projected = self.input_projection(block_input)  # 채널 맞춤
        x_final = x_final + block_input_projected
    
        return x_final


# EfficientMedFormer - AnisotropicConvBlock 기반 새로운 아키텍처 (최종 모델)
class EfficientMedFormer(nn.Module):
    def __init__(self, *, image_size=256, num_classes,
                 channels=3, kernel_size=5):
        super().__init__()

        # AnisotropicConvBlock 기반 계층적 특징 추출 (32×32×48까지)
        self.hierarchical_encoder = HierarchicalEncoder(image_size=image_size, channels=channels)

        # Stage 1 (32×32) - 채널 확장 후 특징 보간
        # GCAA_FormerBlock 1a: 48→64 channels (첫 확장)
        self.gcaa_block1a = GCAA_FormerBlock(
            in_channels=48, out_channels=64,
            kernel_size=kernel_size
        )

        # GCAA_FormerBlock 1b: 64→64 channels (특징 보간)
        self.gcaa_block1b = GCAA_FormerBlock(
            in_channels=64, out_channels=64,
            kernel_size=kernel_size
        )

        # GCAA_FormerBlock 1c: 64→64 channels (특징 보간)
        self.gcaa_block1c = GCAA_FormerBlock(
            in_channels=64, out_channels=64,
            kernel_size=kernel_size
        )

        # AnisotropicConvBlock Downsample 1: 32×32×64 → 16×16×64 (다운샘플링, 중간 해상도용 5×1, 1×5)
        self.aniso_downsample1 = AnisotropicConvBlock(in_channels=64, out_channels=64, downsample=True, kernel_size=5)

        # Stage 2 (16×16) - 채널 확장 후 특징 보간 (중간 해상도용 5×5)
        # GCAA_FormerBlock 2a: 64→96 channels (해상도 변경 후 확장)
        self.gcaa_block2a = GCAA_FormerBlock(
            in_channels=64, out_channels=96,
            kernel_size=5
        )

        # GCAA_FormerBlock 2b: 96→96 channels (특징 보간)
        self.gcaa_block2b = GCAA_FormerBlock(
            in_channels=96, out_channels=96,
            kernel_size=5
        )

        # GCAA_FormerBlock 2c: 96→96 channels (특징 보간)
        self.gcaa_block2c = GCAA_FormerBlock(
            in_channels=96, out_channels=96,
            kernel_size=5
        )

        # GCAA_FormerBlock 2d: 96→96 channels (특징 보간)
        self.gcaa_block2d = GCAA_FormerBlock(
            in_channels=96, out_channels=96,
            kernel_size=5
        )

        # AnisotropicConvBlock Downsample 2: 16×16×96 → 8×8×96 (다운샘플링, 작은 해상도용 3×1, 1×3)
        self.aniso_downsample2 = AnisotropicConvBlock(in_channels=96, out_channels=96, downsample=True, kernel_size=3)

        # Stage 3 (8×8) - 점진적 채널 확장 (작은 해상도용 3×3)
        # GCAA_FormerBlock 3a: 96→128 channels (해상도 변경 후 확장)
        self.gcaa_block3a = GCAA_FormerBlock(
            in_channels=96, out_channels=128,
            kernel_size=3
        )

        # GCAA_FormerBlock 3b: 128→160 channels (중간 확장)
        self.gcaa_block3b = GCAA_FormerBlock(
            in_channels=128, out_channels=160,
            kernel_size=3
        )

        # GCAA_FormerBlock 3c: 160→192 channels (중간 확장)
        self.gcaa_block3c = GCAA_FormerBlock(
            in_channels=160, out_channels=192,
            kernel_size=3
        )

        # GCAA_FormerBlock 3d: 192→256 channels (최종 확장)
        self.gcaa_block3d = GCAA_FormerBlock(
            in_channels=192, out_channels=256,
            kernel_size=3
        )

        # 최종 채널 확장: 256→1024 channels + Dropout
        self.channel_expansion = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
            # nn.Dropout2d(0.15)  # Feature map dropout for regularization
        )

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification head with dropout
        self.classifier = nn.Sequential(
            # nn.Dropout(0.2),  # Linear layer dropout
            nn.Linear(512, num_classes)
        )

    def forward(self, img):
        # AnisotropicConvBlock 기반 계층적 특징 추출
        x = self.hierarchical_encoder(img)  # [B, 48, 32, 32]

        # Stage 1 (32×32): 채널 확장 후 특징 보간
        x = self.gcaa_block1a(x)  # [B, 64, 32, 32] (첫 확장)
        x = self.gcaa_block1b(x)  # [B, 64, 32, 32] (특징 보간)
        x = self.aniso_downsample1(x)  # [B, 64, 16, 16] (다운샘플링)

        # Stage 2 (16×16): 채널 확장 후 특징 보간
        x = self.gcaa_block2a(x)  # [B, 96, 16, 16] (해상도 변경 후 확장)
        x = self.gcaa_block2b(x)  # [B, 96, 16, 16] (특징 보간)
        x = self.aniso_downsample2(x)  # [B, 96, 8, 8] (다운샘플링)

        # Stage 3 (8×8): 점진적 채널 확장
        x = self.gcaa_block3a(x)  # [B, 128, 8, 8] (해상도 변경 후 확장)
        x = self.gcaa_block3b(x)  # [B, 160, 8, 8] (중간 확장)
        x = self.gcaa_block3c(x)  # [B, 192, 8, 8] (중간 확장)
        x = self.gcaa_block3d(x)  # [B, 256, 8, 8] (최종 확장)
        x = self.channel_expansion(x)  # [B, 512, 8, 8] (최종 채널 확장 + Dropout)

        # Global Average Pooling + Classification
        x = self.global_pool(x)  # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        x = self.classifier(x)  # [B, num_classes] (with Dropout)

        return x