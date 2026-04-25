import os, sys, re, yaml, contextlib 
from pathlib import Path 
from functools import partial 
from ..core import register
from ..misc.dist_utils import Multiprocess_sync, is_dist_available_and_initialized    
from ..backbone.common import FrozenBatchNorm2d   

import torch  
import torch.nn as nn
  
from engine.backbone.hgnetv2 import StemBlock, HG_Stage  
from engine.deim.hybrid_encoder import RepNCSPELAN4, StandardConvBlock, ConvNormLayer_fuse, SCDown, CSPLayer, TransformerEncoderBlock
from engine.deim.dfine_decoder import DFINETransformer  

from engine.modules.ultralytics_nn.conv import Concat, Add, Conv    
from engine.modules.ultralytics_nn.block import C2f_Block

from engine.modules.custom_nn.neck_module.mfa import MFA, MFA_StdSCM, StdConvMFA, MFA_NoHighOrder, MFA_NoLowOrder, MFA_NoShortcut, SCM, CSP_SCM, GatedFusion, DownsampleConv
 
RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
  
__all__ = ['HyperDSCNet'] 
    
@register(force=True) # 避免因为导入导致的多次注册    
class HyperDSCNet(nn.Module): 
    __share__ = ['num_classes', 'eval_spatial_size']
    def __init__(self, \
        yaml_path,
        pretrained=None,   
        freeze_stem_only=False,
        freeze_at=-1, 
        freeze_norm=False,
        num_classes=80,
        eval_spatial_size=(640, 640)
    ):     
        super().__init__()   
        d = yaml_load(yaml_path)
        backbone, encoder, decoder, self.save = parse_model(d, ch=3, nc=num_classes, eval_spatial_size=eval_spatial_size, verbose=True)     
        self.backbone = backbone
        self.encoder = encoder 
        self.decoder = decoder   

        # print(self.backbone.state_dict().keys())
  
        if freeze_at >= 0:    
            self._freeze_parameters(self.backbone[0])     
            if not freeze_stem_only: 
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i + 1])    

        if freeze_norm:   
            self._freeze_norm(self.backbone)  
   
        if pretrained:  
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
            try:  
                state = torch.load(pretrained, map_location='cpu')    
                print(f"Loaded stage1 {pretrained} HGNetV2 from local file.")

                # need_keep_key_prefix_list = ['0.*', '1.*']  
                # need_pop_key_list = []    
                # for key in state.keys():
                #     need_pop = True
                #     for keep in need_keep_key_prefix_list:     
                #         if re.match(keep, key):
                #             need_pop = False
                #             break     
                #     if need_pop:     
                #         need_pop_key_list.append(key)
                # for key in need_pop_key_list:   
                #     state.pop(key)
   
                print(RED + f'Loading Pretrained State Dict Key Names:{state.keys()}' + RESET) 
    
                self.backbone.load_state_dict(state, strict=False)
    
            except (Exception, KeyboardInterrupt) as e:
                if (is_dist_available_and_initialized() and torch.distributed.get_rank() == 0) or (not is_dist_available_and_initialized()):     
                    print(f"Loading Backbone Pretrained Weight Error. Message:{str(e)}")
                exit()    
    
    def forward(self, x, targets=None):
        y = []
        for idx, m in enumerate(list(self.backbone.children()) + list(self.encoder.children())):
            # print(idx, m.f, m.i)
            if m.f != -1:  # if not from previous layer  
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers  
            x = m(x)  # run   
            y.append(x if m.i in self.save else None)  # save output 
    
        x = self.decoder([y[j] for j in self.decoder.f], targets)
        return x
  
    def deploy(self, ):
        self.eval()
        for m in self.modules():    
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()    
        return self
    
    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)    
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child) 
        return m 

    def _freeze_parameters(self, m: nn.Module):   
        for p in m.parameters():
            p.requires_grad = False

def yaml_load(file="data.yaml", append_filename=False):    
    """  
    Load YAML data from a file.
    
    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.    

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"     
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string 

        # Remove special characters 
        if not s.isprintable():  
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
    
        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)     
        if append_filename:
            data["yaml_file"] = str(file)
        return data
 
def parse_module(d, i, f, m, args, ch, nc=None, eval_spatial_size=None):   
    import ast
    try:
        if m == 'node_mode':
            m = d[m]  
            if len(args) > 0:    
                if args[0] == 'head_channel':   
                    args[0] = int(d[args[0]])
        t = m    
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
    except: 
        pass
    
    selfatt, selfatt_args_index = None, -1
    if type(args) is list:    
        for j, a in enumerate(args):   
            if isinstance(a, str):   
                with contextlib.suppress(ValueError):
                    try:  
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                    except:     
                        args[j] = a   
            elif type(a) is dict:
                if 'module' in a: 
                    module_ = a['module']
                    try:
                        module_ = getattr(torch.nn, module_[3:]) if 'nn.' in module_ else globals()[module_]
                    except Exception as e:    
                        raise Exception(f'{module_} is maybe not import in task.py, please check. {e}')    
                    module_param = a.get('param', {})
                    for k in module_param:   
                        p = module_param[k]
                        try:     
                            module_param[k] = locals()[p] if p in locals() else (getattr(torch.nn, p[3:]) if 'nn.' in p else ast.literal_eval(p)) 
                        except:
                            module_param[k] = p 
                    args[j] = partial(module_, **module_param)
                if 'selfatt' in a:
                    selfatt = a['selfatt']    
                    selfatt_args_index = j  
    if selfatt_args_index != -1:   
        args.pop(j) 
    
    c2 = ch[-1]     
    if m in {StemBlock, HG_Stage}:  
        c1, cmid, c2 = ch[f], args[0], args[1]
        args = [c1, cmid, c2, *args[2:]]
    elif m in {RepNCSPELAN4, StandardConvBlock, CSPLayer, ConvNormLayer_fuse, SCDown}:
        c1, c2 = ch[f], args[0]
        args = [c1, c2, *args[1:]]
    elif m in {TransformerEncoderBlock}:  
        c2 = ch[f] 
        args = [c2, *args]
    elif m is Concat:
        c2 = sum(ch[x] for x in f)
    elif m in {DownsampleConv}: 
        c2 = ch[f]     
        args = [c2, *args]
    elif m in {Conv}:    
        c1, c2 = ch[f], args[0]
        args = [c1, c2, *args[1:]]    
    elif m in {C2f_Block}: 
        c1, c2 = ch[f], args[0]
        if selfatt is not None:
            args = ([c1, c2, *args[1:]], {'selfatt':selfatt})   
        else:     
            args = [c1, c2, *args[1:]]
    elif m in {SCM}:    
        c2 = ch[f]   
        args = [c2, *args]
    elif m is CSP_SCM:
        c1, c2 = ch[f], args[0]
        args = [c1, c2, *args[1:]]
    elif m in {MFA, MFA_StdSCM, StdConvMFA, MFA_NoHighOrder, MFA_NoLowOrder, MFA_NoShortcut}:
        c1 = ch[f[1]]     
        c2 = args[0]   
        args = [c1, c2, *args[1:]]    
    elif m is GatedFusion:   
        c2 = ch[f[0]]    
    elif m is Add:   
        c2 = ch[f[-1]]    
    elif m is DFINETransformer: 
        args["feat_channels"] = [ch[x] for x in f]     
        args["num_classes"] = nc  
        args["eval_spatial_size"] = eval_spatial_size
    else:  
        c2 = ch[f]   
    
    if type(args) is dict:   
        m_ = m(**args)   
    elif type(args) is list:
        m_ = m(*args)  # module    
    else:
        m_ = m(*args[0], **args[1])   
    t = str(m)[8:-2].replace('__main__.', '')  # module type
    m_.np = sum(x.numel() for x in m_.parameters())  # number params
    m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type

    return m_, c2, t, args

def parse_model(d, ch, nc, eval_spatial_size, verbose=True):   
    if verbose:    
        print(ORANGE + f"{'':>3}{'from':>10}{'params':>10}  {'module':<60}{'arguments':<30}" + RESET) 
    layer_index, ch = 0, [ch]
    backbone_layers, encoder_layers, decoder_model, save, c2 = [], [], None, [], ch[-1]  # layers, savelist, ch out  
  
    if verbose:
        print(BLUE + "-"*40 + "BackBone" + "-"*40 + RESET)
    for f, m, args in d["backbone"]:  
     
        m_, c2, t, args = parse_module(d, layer_index, f, m, args, ch)   

        if verbose:
            print(ORANGE + f"{layer_index:>3}{str(f):>10}{m_.np:10.0f}  {t:<60}{str(args):<30}" + RESET)  # print
        
        save.extend(x % layer_index for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        backbone_layers.append(m_)    
        if layer_index == 0:   
            ch = []    
        ch.append(c2)
    
        layer_index += 1   
    
    if verbose:
        print(BLUE + "-"*40 + "Enchoder" + "-"*40 + RESET) 
    for f, m, args in d["encoder"]:
        
        m_, c2, t, args = parse_module(d, layer_index, f, m, args, ch)    
     
        if verbose:
            print(ORANGE + f"{layer_index:>3}{str(f):>10}{m_.np:10.0f}  {t:<60}{str(args):<30}" + RESET)  # print
        
        save.extend(x % layer_index for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist   
        encoder_layers.append(m_)   
        ch.append(c2)
   
        layer_index += 1 
 
    if verbose:
        print(BLUE + "-"*40 + "Decoder" + "-"*40 + RESET)
    for f, m, args in d["decoder"]:
        
        m_, c2, t, args = parse_module(d, layer_index, f, m, args, ch, nc, eval_spatial_size)     

        if verbose:
            print(ORANGE + f"{layer_index:>3}{str(f):>10}{m_.np:10.0f}  {t:<60}{str(args):<30}" + RESET)  # print     
        
        save.extend(x % layer_index for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        decoder_model = m_
        ch.append(c2)
    
    # print(ch)   
    return nn.Sequential(*backbone_layers), nn.Sequential(*encoder_layers), decoder_model, sorted(save)