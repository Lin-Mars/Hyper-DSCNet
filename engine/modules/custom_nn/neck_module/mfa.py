import torch, math
import torch.nn as nn
import torch.nn.functional as F

from engine.modules.ultralytics_nn.conv import Conv
from engine.modules.ultralytics_nn.block import C3, C2f

__all__ = ['RCSEConv', 'RCSE_C3k2', 'MFA', 'MFA_StdSCM', 'StdConvMFA',
           'MFA_NoHighOrder', 'MFA_NoLowOrder', 'MFA_NoShortcut',
           'SCM', 'CSP_SCM', 'DownsampleConv', 'GatedFusion']    
 
class RCSEConv(nn.Module):  
    """Residual Channel-Spatial Enhancement Convolution (depthwise separable)."""
    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, bias=False):  
        super().__init__()    
        if p is None:   
            p = (d * (k - 1)) // 2    
        self.dw = nn.Conv2d(    
            c_in, c_in, kernel_size=k, stride=s,  
            padding=p, dilation=d, groups=c_in, bias=bias     
        )
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()
 
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(self.bn(x))

class RCSEBottleneck(nn.Module):
    """RCSE bottleneck block using depthwise separable convolutions."""  
    def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
        super().__init__()   
        c_ = int(c2 * e)    
        self.cv1 = RCSEConv(c1, c_, k1, s=1, p=None, d=1)   
        self.cv2 = RCSEConv(c_, c2, k2, s=1, p=None, d=d2) 
        self.add = shortcut and c1 == c2  
     
    def forward(self, x):   
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class RCSE_C3k(C3):
    """C3-style block using RCSE bottleneck for low-order feature extraction."""
    def __init__(
        self,     
        c1,                
        c2,  
        n=1,   
        shortcut=True,  
        g=1,  
        e=0.5,              
        k1=3,  
        k2=5,               
        d2=1   
    ):    
        super().__init__(c1, c2, n, shortcut, g, e)   
        c_ = int(c2 * e)  

        self.m = nn.Sequential(
            *(
                RCSEBottleneck(   
                    c_, c_,     
                    shortcut=shortcut,
                    e=1.0,
                    k1=k1,
                    k2=k2,     
                    d2=d2
                )
                for _ in range(n)  
            )
        )

class RCSE_C3k2(C2f):   
    """C2f-style block using RCSE for low-order feature extraction in MFA."""  
    def __init__(
        self,
        c1, 
        c2,     
        n=1,          
        dsc3k=False,  
        e=0.5,  
        g=1,   
        shortcut=True,
        k1=3,       
        k2=7,  
        d2=1         
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        if dsc3k:     
            self.m = nn.ModuleList(     
                RCSE_C3k(
                    self.c, self.c,  
                    n=2,           
                    shortcut=shortcut,     
                    g=g,
                    e=1.0,    
                    k1=k1,
                    k2=k2,   
                    d2=d2    
                ) 
                for _ in range(n)   
            )
        else:
            self.m = nn.ModuleList(    
                RCSEBottleneck(
                    self.c, self.c,
                    shortcut=shortcut,    
                    e=1.0,  
                    k1=k1,     
                    k2=k2,   
                    d2=d2
                )
                for _ in range(n)  
            )  
 
class AdaptiveHyperedgeGenerator(nn.Module):
    """Generates adaptive hyperedge participation matrix via context-aware prototype generation."""    
    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1, context="both"):
        super().__init__()    
        self.num_heads = num_heads   
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads
        self.context = context

        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim)) 
        nn.init.xavier_uniform_(self.prototype_base)    
        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, num_hyperedges * node_dim)  
        elif context == "both": 
            self.context_net = nn.Linear(2*node_dim, num_hyperedges * node_dim)     
        else:
            raise ValueError(  
                f"Unsupported context '{context}'. "     
                "Expected one of: 'mean', 'max', 'both'."
            )    

        self.pre_head_proj = nn.Linear(node_dim, node_dim)     
    
        self.dropout = nn.Dropout(dropout)   
        self.scaling = math.sqrt(self.head_dim)
  
    def forward(self, X):
        B, N, D = X.shape
        if self.context == "mean":   
            context_cat = X.mean(dim=1)          
        elif self.context == "max":
            context_cat, _ = X.max(dim=1)          
        else:
            avg_context = X.mean(dim=1)     
            max_context, _ = X.max(dim=1)   
            context_cat = torch.cat([avg_context, max_context], dim=-1) 
        prototype_offsets = self.context_net(context_cat).view(B, self.num_hyperedges, D)  
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets 
 
        X_proj = self.pre_head_proj(X) 
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)     
   
        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling 
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1) 
 
        logits = self.dropout(logits)  

        return F.softmax(logits, dim=1)

class HypergraphConv(nn.Module):    
    """Two-stage hypergraph convolution: vertex-to-edge aggregation and edge-to-vertex dissemination."""   
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, dropout=0.1, context="both"):
        super().__init__()  
        self.edge_generator = AdaptiveHyperedgeGenerator(embed_dim, num_hyperedges, num_heads, dropout, context)    
        self.edge_proj = nn.Sequential( 
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )
        self.node_proj = nn.Sequential( 
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()     
        ) 
        
    def forward(self, X):
        A = self.edge_generator(X)  
   
        He = torch.bmm(A.transpose(1, 2), X)     
        He = self.edge_proj(He)
        
        X_new = torch.bmm(A, He)  
        X_new = self.node_proj(X_new)
        
        return X_new + X
  
class SCM(nn.Module):
    """Syndromic Context Modeling (SCM) module.
    Applies adaptive hypergraph convolution to 4D feature maps for high-order semantic association modeling."""    
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=8, dropout=0.1, context="both"):
        super().__init__()   
        self.embed_dim = embed_dim  
        self.hgnn = HypergraphConv(    
            embed_dim=embed_dim,  
            num_hyperedges=num_hyperedges,
            num_heads=num_heads, 
            dropout=dropout,
            context=context
        )
     
    def forward(self, x): 
        B, C, H, W = x.shape  
        tokens = x.flatten(2).transpose(1, 2)   
        tokens = self.hgnn(tokens) 
        x_out = tokens.transpose(1, 2).view(B, C, H, W)
        return x_out    

class CSP_SCM(nn.Module):
    """CSP-style block integrating SCM for high-order branch in MFA."""   
    def __init__(self, c1, c2, e=1.0, num_hyperedges=8, context="both"):   
        super().__init__()
        c_ = int(c2 * e) 
        assert c_ % 16 == 0, "Dimension of SCM should be a multiple of 16."   
        num_heads = c_ // 16     
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = SCM(embed_dim=c_, 
                          num_hyperedges=num_hyperedges, 
                          num_heads=num_heads,
                          dropout=0.1,
                          context=context)
        self.cv3 = Conv(2 * c_, c2, 1)  
        
    def forward(self, x): 
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)) 

class MultiScaleAlign(nn.Module):
    """Aligns and fuses multi-scale feature maps to a common spatial resolution."""     
    def __init__(self, c_in, channel_adjust):   
        super(MultiScaleAlign, self).__init__()  
        self.downsample = nn.AvgPool2d(kernel_size=2)    
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        if channel_adjust: 
            self.conv_out = Conv(4 * c_in, c_in, 1)     
        else:    
            self.conv_out = Conv(3 * c_in, c_in, 1)   

    def forward(self, x):    
        x1_ds = self.downsample(x[0])
        x3_up = self.upsample(x[2])    
        x_cat = torch.cat([x1_ds, x[1], x3_up], dim=1)
        out = self.conv_out(x_cat)
        return out

class MFA(nn.Module):   
    """Multi-finding Adaptive Fusion (MFA) module.
    Parallel multi-branch architecture: high-order branches (CSP_SCM) for syndromic context,
    low-order branch (RCSE) for local features, and a shortcut branch."""
    def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True, shortcut=False, e1=0.5, e2=1, context="both", channel_adjust=False):   
        super().__init__()     
        self.c = int(c2 * e1)     
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1) 
        self.m = nn.ModuleList(  
            RCSE_C3k(self.c, self.c, 2, shortcut, k1=3, k2=7) if dsc3k else RCSEBottleneck(self.c, self.c, shortcut=shortcut) for _ in range(n)  
        )     
        self.fuse = MultiScaleAlign(c1, channel_adjust)
        self.branch1 = CSP_SCM(self.c, self.c, e2, num_hyperedges, context)   
        self.branch2 = CSP_SCM(self.c, self.c, e2, num_hyperedges, context)    
                    
    def forward(self, X): 
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1)) 
        out1 = self.branch1(y[1])    
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)   
        y[1] = out1     
        y.append(out2)
        return self.cv2(torch.cat(y, 1))   
  
class StdConvSCM(nn.Module):
    """Capacity-matched standard Conv replacement for SCM.
    Conv3x3 -> Conv3x3 -> Conv1x1 with residual, ~313,736 params (matches SCM's 313,728).
    """
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, 137, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(137)
        self.conv2 = nn.Conv2d(137, 114, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(114)
        self.conv3 = nn.Conv2d(114, c, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.act(self.bn3(self.conv3(out)))
        return out + x


class CSP_StdConv(nn.Module):
    """CSP block with SCM replaced by capacity-matched standard convolutions."""
    def __init__(self, c1, c2, e=1.0, num_hyperedges=8, context="both"):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = StdConvSCM(c_)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class MFA_StdSCM(nn.Module):
    """MFA with SCM replaced by capacity-matched standard convolutions (ablation)."""
    def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True, shortcut=False, e1=0.5, e2=1, context="both", channel_adjust=False):
        super().__init__()
        self.c = int(c2 * e1)
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            RCSE_C3k(self.c, self.c, 2, shortcut, k1=3, k2=7) if dsc3k else RCSEBottleneck(self.c, self.c, shortcut=shortcut) for _ in range(n)
        )
        self.fuse = MultiScaleAlign(c1, channel_adjust)
        self.branch1 = CSP_StdConv(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = CSP_StdConv(self.c, self.c, e2, num_hyperedges, context)

    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)
        return self.cv2(torch.cat(y, 1))


class StdConvMFA(nn.Module):
    """Capacity-matched standard convolution replacement for the full MFA (ablation)."""
    def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True, shortcut=False,
                 e1=0.5, e2=1, context="both", channel_adjust=False):
        super().__init__()
        self.fuse = MultiScaleAlign(c1, channel_adjust)
        self.conv1 = nn.Conv2d(c1, 235, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(235)
        self.conv2 = nn.Conv2d(235, 227, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(227)
        self.conv3 = nn.Conv2d(227, c2, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, X):
        x = self.fuse(X)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        return x


class MFA_NoHighOrder(nn.Module):
    """MFA with high-order branches removed (ablation). Keeps shortcut + low-order."""
    def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True, shortcut=False,
                 e1=0.5, e2=1, context="both", channel_adjust=False):
        super().__init__()
        self.c = int(c2 * e1)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            RCSE_C3k(self.c, self.c, 2, shortcut, k1=3, k2=7) if dsc3k else RCSEBottleneck(self.c, self.c, shortcut=shortcut) for _ in range(n)
        )
        self.fuse = MultiScaleAlign(c1, channel_adjust)

    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MFA_NoLowOrder(nn.Module):
    """MFA with low-order branch removed (ablation). Keeps shortcut + high-order."""
    def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True, shortcut=False,
                 e1=0.5, e2=1, context="both", channel_adjust=False):
        super().__init__()
        self.c = int(c2 * e1)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(3 * self.c, c2, 1)
        self.fuse = MultiScaleAlign(c1, channel_adjust)
        self.branch1 = CSP_SCM(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = CSP_SCM(self.c, self.c, e2, num_hyperedges, context)

    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(2, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y[1] = out1
        y.append(out2)
        return self.cv2(torch.cat(y, 1))


class MFA_NoShortcut(nn.Module):
    """MFA with shortcut branch removed (ablation). Keeps high-order + low-order."""
    def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True, shortcut=False,
                 e1=0.5, e2=1, context="both", channel_adjust=False):
        super().__init__()
        self.c = int(c2 * e1)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            RCSE_C3k(self.c, self.c, 2, shortcut, k1=3, k2=7) if dsc3k else RCSEBottleneck(self.c, self.c, shortcut=shortcut) for _ in range(n)
        )
        self.fuse = MultiScaleAlign(c1, channel_adjust)
        self.branch1 = CSP_SCM(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = CSP_SCM(self.c, self.c, e2, num_hyperedges, context)

    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(2, 1))
        out1 = self.branch1(y[0])
        out2 = self.branch2(y[0])
        y[0] = out1
        y.extend(m(y[-1]) for m in self.m)
        y.append(out2)
        return self.cv2(torch.cat(y, 1))


class DownsampleConv(nn.Module):   
    """Downsampling block with optional channel adjustment."""
    def __init__(self, in_channels, channel_adjust=False):  
        super().__init__() 
        self.downsample = nn.AvgPool2d(kernel_size=2)
        if channel_adjust:
            self.channel_adjust = Conv(in_channels, in_channels * 2, 1)    
        else:
            self.channel_adjust = nn.Identity() 

    def forward(self, x):
        return self.channel_adjust(self.downsample(x))

class GatedFusion(nn.Module):
    """Gated residual fusion: output = x_original + gate * x_enhanced."""
    def __init__(self):  
        super().__init__()   
        self.gate = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):    
        out = x[0] + self.gate * x[1]
        return out     

