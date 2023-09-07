
#Copyright(c) CL.Gao All rights reserved

import copy
from abc import ABCMeta


import torch
from torch import nn
import torch.distributed as dist
from torch.nn import functional as F

class BaseModule(nn.Module,metaclass=ABCMeta):
    """ Args:
        init_cfg (dict, optional): Initialization config dict.
    """
    def __init__(self, init_cfg=None):
        super(BaseModule,self).__init__()
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

    @property
    def is_init(self):
        return self._is_init
    def init_weights(self):
        """initialize the weights. """
        is_top_level_module=False
        if not hasattr(self, '_params_init_info'):
            self._params_init_info=defaultdict(dict)
            is_top_level_module=True
            for name, param in self.named _parameters():
                self._params_init_info[param]['init_info']=f'The value is the same before and '\
                                                           f'after calling `init_weights` '\
                                                           f'of {self.__class__.__name__}'
                self._params_init_info[param]['tmp_mean_value']=param.data.mean()

            for sub_module in self.modules():
                sub_module._params_init_info=self._params_init_info
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_name else 'NNModel'

        from ..cnn import initialize
        from ..cnn.utils.weight_init import update_init_info
        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(f'initialize{module_name} with init_cfg {self.init_cfg}',logger=logger_name)
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    if self.init_cfg['type']=='Pretrained':
                        return 
            for m in self.children():
                if hasattr(m, 'init_weights'):
                    m.init_weights()
                    update_init_info(m,init_info=f'initialized by user defined `init_weights`'
                                                 f' in {m.__class__.__name__}')
            self._is_init=True
        else:
            warning.warn(f'init_weights of {self.__class__.__name__} has been called more than once.')

        if is_top_level_module:
            self._dump_init_info(logger_name)
            for sub_module in self.modules():
                del sub_module._params_init_info
    
    def _dump_init_info(self, logger_name):
        """
        Args:
            logger_name (str): The name of logger.
        """
        logger = get_logger(logger_name)
        with_file_handler=False
        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                handler.stream.write('name of parameter - initialization information \n')
                for name, param in self.named_parameters():
                    handler.stream.write(f'\n{name}-"{param.shape} {self._params_init_info[param]['init_info']}')
                handler.stream.flush()
                with_file_handler=True
        if not with_file_handler:
            for name, param in self.named_parameters():
                print_log(f'\n{name}-{param.shape}: {self._params_init_info[param]['init_info']}',logger=logger_name)

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s+=f'\ninit_cfg={self.init_cfg}'
        return s
    
class Base3DFusionModel(BaseModule, metaclass=ABCMeta):
    def __init__(self, init_cfg=None):
        super(Base3DFusionMode,self).__init__(init_cfg)
        self.fp16_enabled=False
    def _parse_losses(self, losses):
        log_vars=OrdereDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name]=loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name]=sum(_loss.mean()for _loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")
        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)
        log_vars["loss"]=loss
        for loss_name, loss_value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value= loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name]=loss_value.item()
        return loss, log_vars
    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars= self._parse_losses(losses)
        outputs=dict(loss=loss, log_vars=log_vars, num_samples=len(data["metas"]))
        return outputs
    def val_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars= self._parse_losses(losses)
        outputs=dict(loss=loss, log_vars=log_vars, num_samples=len(data["metas"]))
        return outputs
    

class BEVFusion(Base3DFusionModel):
    def __init__(self,encoders: Dict[str,Any], decoder: Dict[str,Any],heads: Dict[str, Any],**kwargs, ) -> None:
        super(BEVFusion,self).__init__()
        self.encoders=nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"]=nn.ModuleDict(
                {
                    "backbone":build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points",-1)>0:
                voxelize_module=Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module=DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"]=nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser=_None
        self.decoder = nn.MOduleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads=nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name]=build_head(heads[name])
        if "loss_scale" in kwargs:
            self.loss_scale=kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name]=1.0
        self.init_weights()
    def init_weights(self) ->None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()
    def extract_camera_features(self,x,points,camera2ego,lidar2ego,lidar2camera,lidar2image,camera_intrinsics,camera2lidar,img_aug_matrix,lidar_aug_matrix,img_metas,) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x=x.view(B*N,C,H,W)

        x=self.encoders["camera"]["backbone"](x)
        x=self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x=x[0]
        BN, C, H, W = x.size()
        x=x.view(B, int(BN/B), C, H, W)
        
        x=self.encoders["camera"]["vtransform"](x,points,camea2ego,lidar2ego,lidar2image,camera_intrinsics,camera2lidar,img_aug_matrix,lidar_aug_matrix,img_metas,)
        return x
    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1,0]+1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x
    
    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes=[], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret)==3:
                f, c, n=ret
            else:
                assert len(ret)==2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c,(1,0),mode="constant", value=k))
            if n is not None:
                sizes.append(n)
        
        feats = torch.cat(fats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes)>0:
            sizes= torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False)/sizes.type_as(feats).view(-1,1)
                feats=feats.contiguous()
        return feats, coords, sizes
    def forward(self, img, points, camera2ego, lidar2ego, lidar2camera,lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix,metas, gt_masks_bev=None, gt_bboxes_3d=None,gt_labels_3d=None,**kwargs,):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            output = self.forward_single(img,points, camera2ego, lidar2ego, lidar2camera, lidar2image, camera_intrinsics, camera2lidar, img_augh_matrix, lidar_aug_matrix, metas, gt_masks_bev, gt_bboxex_ed, gt_labels_3d,**kwargs,)

    def forward_single(self, img, points, camera2ego, lidar2ego, lidar2camera, lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix, metas, gt_masks_bev=None, gt_bboxes_3d=None, gt_labels_3d=None,**kwargs,):
        features=[]
        for sensor in (self.encoders if self.training else list(self.encoders.keys())[::-1]):
            if sensor=="camera":
                feature = self.extract_camera_features(img, points, camera2ego, lidar2ego, lidar2camera, lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix, metas,)
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            featuresd = features[::-1]
        if self.fuser is not None:
            x=self.fuser(features)
        else:
            assert len(features)==1, features
            x = features[0]
        
        batch_size = x.shape[0]

        x=self.decoder["backbone"](x)
        x=self.decoder["neck"](x)

        if self.training:
            outputs={}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict=head(x,metas)
                    losses=head.loss(gt_bboxes_3d, gt_labels_ed, pred_dict)
                elif type == "map":
                    losses=head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val*self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"]=val
            return outputs
        else:
            outputs=[{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type== "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxex):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu")
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type=="map":
                    logits=head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev":logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
        




        


