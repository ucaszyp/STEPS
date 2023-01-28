import os.path as osp

import pytorch_lightning
import torch.nn.functional as F
from mmcv import Config
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import numpy as np

from components import freeze_model, unfreeze_model, ImagePool, get_smooth_loss
from utils import EWMA
from .disp_net import DispNet
from .gan import GANLoss, NLayerDiscriminator
from .layers import SSIM, Backproject, Project
from .registry import MODELS
from .utils import *
from SCI.model import *
from transforms import EqualizeHist

def build_disp_net(option, check_point_path):
    # create model
    model: pytorch_lightning.LightningModule = MODELS.build(name=option.model.name, option=option)
    model.load_state_dict(torch.load(check_point_path, map_location='cpu')['state_dict'])
    model.freeze()
    model.eval()

    # return
    return model


@MODELS.register_module(name='rnw')
class RNWModel(LightningModule):
    """
    The training process
    """
    def __init__(self, opt):

        super(RNWModel, self).__init__()
        self.test = opt.test
        self.opt = opt.model
        self._equ_limit = self.opt.equ_limit
        self._to_tensor = ToTensor()
        # self.epoch = 0
        self.predictions = {}
        self.gt = {}
        self.min_depth = 1e-5
        self.max_depth = 60.0

        # components
        self.gan_loss = GANLoss('lsgan')
        self.image_pool = ImagePool(50)
        self.ssim = SSIM()
        self.backproject = Backproject(self.opt.imgs_per_gpu, self.opt.height, self.opt.width)
        self.project_3d = Project(self.opt.imgs_per_gpu, self.opt.height, self.opt.width)
        self.ego_diff = EWMA(momentum=0.98)

        # networks
        self.S = Network(stage=3)
        self.G = DispNet(self.opt)
        in_chs_D = 3 if self.opt.use_position_map else 1
        self.D = NLayerDiscriminator(in_chs_D, n_layers=3)

        # init SCI_Model
        self.S.enhance.in_conv.apply(self.S.weights_init)
        self.S.enhance.conv.apply(self.S.weights_init)
        self.S.enhance.out_conv.apply(self.S.weights_init)
        self.S.calibrate.in_conv.apply(self.S.weights_init)
        self.S.calibrate.convs.apply(self.S.weights_init)
        self.S.calibrate.out_conv.apply(self.S.weights_init)


        # register image coordinates
        if self.opt.use_position_map:
            h, w = self.opt.height, self.opt.width
            height_map = torch.arange(h).view(1, 1, h, 1).repeat(1, 1, 1, w) / (h - 1)
            width_map = torch.arange(w).view(1, 1, 1, w).repeat(1, 1, h, 1) / (w - 1)

            self.register_buffer('height_map', height_map, persistent=False)
            self.register_buffer('width_map', width_map, persistent=False)

        # build day disp net
        self.day_dispnet = build_disp_net(
            Config.fromfile(osp.join('configs/', f'{self.opt.day_config}.yaml')),
            self.opt.day_check_point
        )

        # link to dataset
        self.data_link = opt.data_link

        # manual optimization
        self.automatic_optimization = False

    def forward(self, inputs):
        if not self.test:
            return self.G(inputs)
        else:
            self.S.eval()
            sci_gray = inputs[("color_gray", 0, 0)][0].unsqueeze(0)
            sci_color = inputs[("color", 0, 0)][0].unsqueeze(0)                    
            illu_list, _, _, _, _ = self.S(sci_gray)
            illu = illu_list[0][0][0]

            illu = torch.stack([illu, illu, illu])

            illu = illu.unsqueeze(0)
            r = sci_color / illu
            r = torch.clamp(r, 0, 1)
            inputs[("color_aug", 0, 0)] = r
            return self.G(inputs)

    def generate_gan_outputs(self, day_inputs, outputs):
        # (n, 1, h, w)
        night_disp = outputs['disp', 0, 0]
        # if self.current_epoch <= 15:
        with torch.no_grad():
            day_disp = self.day_dispnet(day_inputs)['disp', 0, 0]
            day_disp = F.interpolate(day_disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        # remove scale
        night_disp = night_disp / night_disp.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
        day_disp = day_disp / day_disp.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
        # image coordinates
        if self.opt.use_position_map:
            n = night_disp.shape[0]
            height_map = self.height_map.repeat(n, 1, 1, 1)
            width_map = self.width_map.repeat(n, 1, 1, 1)
        else:
            height_map = None
            width_map = None
        # return
        return day_disp, night_disp, height_map, width_map

    def compute_G_loss(self, night_disp, height_map, width_map):
        G_loss = 0.0
        #
        # Compute G loss
        #
        freeze_model(self.D)
        if self.opt.use_position_map:
            fake_day = torch.cat([height_map, width_map, night_disp], dim=1)
        else:
            fake_day = night_disp
        G_loss += self.gan_loss(self.D(fake_day), True)

        return G_loss

    def compute_D_loss(self, day_disp, night_disp, height_map, width_map):
        D_loss = 0.0
        #
        # Compute D loss
        #
        unfreeze_model(self.D)
        if self.opt.use_position_map:
            real_day = torch.cat([height_map, width_map, day_disp], dim=1)
            fake_day = torch.cat([height_map, width_map, night_disp.detach()], dim=1)
        else:
            real_day = day_disp
            fake_day = night_disp.detach()
        # query
        fake_day = self.image_pool.query(fake_day)
        # compute loss
        D_loss += self.gan_loss(self.D(real_day), True)
        D_loss += self.gan_loss(self.D(fake_day), False)

        return D_loss

    def training_step(self, batch_data, batch_idx):
        # optimizers
        optim_G, optim_D = self.optimizers()

        # tensorboard logger
        logger = self.logger.experiment

        # get input data
        day_inputs = batch_data['day']
        night_inputs = batch_data['night']
        
        # TODO: get relight img
        night_inputs, sci_loss_dict = self.get_sci_relight(night_inputs)
        if self.opt.use_equ:
            night_inputs = self.get_mcie_relight(night_inputs)
        
        # # outputs of G
        outputs = self.G(night_inputs)

        # loss for ego-motion
        disp_loss_dict = self.compute_disp_losses(night_inputs, outputs)
        
        # generate outputs for gan
        day_disp, night_disp, height_map, width_map = self.generate_gan_outputs(day_inputs, outputs)
        
        #
        # optimize G
        #
        # compute loss
        G_loss = self.compute_G_loss(night_disp, height_map, width_map)        
        S_loss = sum(sci_loss_dict.values())
        disp_loss = sum(disp_loss_dict.values())

        # log
        logger.add_scalar('train/disp_loss', disp_loss, self.global_step)
        logger.add_scalar('train/G_loss', G_loss, self.global_step)
        # logger.add_scalar('train/S_loss', S_loss, self.global_step)
        # if self.local_rank == 0:
        #     wandb_data = {"disp-loss": disp_loss,
        #                 "G_loss": G_loss,
        #                 "S_loss": S_loss}
        #     wandb.log(wandb_data)

        # optimize G
        G_loss = G_loss * self.opt.G_weight + disp_loss + S_loss * self.opt.S_weight
        # G_loss = G_loss * self.opt.G_weight + disp_loss
        
        optim_G.zero_grad()
        self.manual_backward(G_loss)
        optim_G.step()


        # optimize D
        #
        # compute loss
        D_loss = self.compute_D_loss(day_disp, night_disp, height_map, width_map)

        # # log
        logger.add_scalar('train/D_loss', D_loss, self.global_step)

        D_loss = D_loss * self.opt.D_weight

        # optimize D
        optim_D.zero_grad()
        self.manual_backward(D_loss)
        optim_D.step()

    def validation_step(self, batch_data, batch_idx):
        
        with torch.no_grad():
            test_input = batch_data
            self.S.eval()
            sci_gray = test_input[("color_gray", 0, 0)]
            sci_color = test_input[("color", 0, 0)]
            gt = test_input[("gt", 0, 0)][0][0]
            gh, gw = gt.shape
            b, c, rh, rw = sci_color.shape                   
            sf = float(gh / rh)

            illu_list, _, _, _, i_k = self.S(sci_gray)
            illu = illu_list[0][0][0]
            illu = torch.stack([illu, illu, illu])
            illu = illu.unsqueeze(0)
            r = sci_color / illu
            r = torch.clamp(r, 0, 1)
            test_input[("color_aug", 0, 0)] = r
            
            disp = self.G(test_input)[("disp", 0, 0)]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            depth = F.interpolate(depth, scale_factor=sf, mode='nearest')
            depth = depth[0, 0, :, :]
            
            mask = (gt > self.min_depth) & (gt < self.max_depth)
            # get values

            pred_vals, gt_vals = depth[mask], gt[mask]
            # compute scale
            scale = torch.median(gt_vals) / torch.median(pred_vals)
            pred_vals *= scale
            pred_vals = torch.clamp(pred_vals, min=self.min_depth, max=self.max_depth)
            # compute error
            error = self.compute_metrics(pred_vals, gt_vals)
            error["img_index"] = test_input["file_name"]
                
        return error


    def training_epoch_end(self, outputs):
        """
        Step lr scheduler
        :param outputs:
        :return:
        """
        sch_G, sch_D = self.lr_schedulers()

        sch_G.step()
        sch_D.step()

        self.data_link.when_epoch_over()


    def validation_epoch_end(self, val_step_outputs):
        logger = self.logger.experiment
        errors = {'abs_rel': [], 'sq_rel': [], 'rmse': [], 'rmse_log': [], 'a1': [], 'a2': [], 'a3': []}
        for out in val_step_outputs:
            for k in errors:
                errors[k].append(out[k])

        errors = {k: sum(v).item() / len(v) for k, v in errors.items()}
        for k, v in errors.items():
            logger.add_scalar(k, v, self.current_epoch)

    def compute_metrics(self, pred, gt):
        thresh = torch.maximum((gt / pred), (pred / gt))
        a1 = ((thresh < 1.25).type(torch.float)).mean()
        a2 = ((thresh < 1.25 ** 2).type(torch.float)).mean()
        a3 = ((thresh < 1.25 ** 3).type(torch.float)).mean()

        rmse = (gt - pred) ** 2
        rmse = torch.sqrt(rmse.mean())
        rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
        rmse_log = torch.sqrt(rmse_log.mean())

        abs_rel = torch.mean(torch.abs(gt - pred) / gt)
        sq_rel = torch.mean(((gt - pred) ** 2) / gt)

        result = {'abs_rel': abs_rel, 'sq_rel': sq_rel, 'rmse': rmse, 'rmse_log': rmse_log, 'a1': a1, 'a2': a2, 'a3': a3}
        return result

    def configure_optimizers(self):
        optim_params = [
            {'params': self.G.parameters(), 'lr': self.opt.G_learning_rate},
            {'params': self.S.parameters(), 'lr': self.opt.S_learning_rate, 'betas': (0.9, 0.999), 'weight_decay': 3e-4}
        ]
        optim_G = Adam(optim_params)
        optim_D = Adam(self.D.parameters(), lr=self.opt.G_learning_rate)

        sch_G = MultiStepLR(optim_G, milestones=[15], gamma=0.5)
        sch_D = MultiStepLR(optim_D, milestones=[15], gamma=0.5)

        return [optim_G, optim_D], [sch_G, sch_D]
    
    def get_sci_relight(self, inputs):

        loss_dict = {}
        src_colors = inputs[('color', 0, 0)]
        b, _, h, w = src_colors.shape
        
        # get sci_color 
        for scale in self.opt.scales:

            for frame_id in self.opt.frame_ids:
                sci_gray = inputs[("color_gray", frame_id, scale)]
                sci_color = inputs[("color", frame_id, scale)]
                loss, illu_list, i_k = self.S._loss(sci_gray, frame_id) #todo: index 0 loss, index 0, -1, 1 img
                nn.utils.clip_grad_norm_(self.S.parameters(), 5)
                if frame_id == 0:
                    loss_dict[("sci_loss", 0, scale)] = loss / len(self.opt.scales)
                
                illu = illu_list[0]
                
                if scale == 0 and frame_id == 0:
                    if self.opt.use_illu_mask:
                        # illu_mask = self.get_illu_mask(illu)
                        illu_mask, k = self.auto_illu_mask(i_k)
                        inputs[("scale_k", 0, 0)] = k.detach()
                        inputs[("light_mask", frame_id, scale)] = illu_mask.detach()
                    inputs[("gray_aug", frame_id, scale)] = (sci_gray / illu).clamp(max=1, min=0)
                
                illu = torch.cat((illu, illu, illu), dim=1)
                r = sci_color / illu
                r = torch.clamp(r, 0, 1)
                inputs[("color_aug", frame_id, scale)] = r  
        
        return inputs, loss_dict
    
    def get_mcie_relight(self, inputs):
        
        src_colors = inputs[('color_aug', 0, 0)]
        b, _, h, w = src_colors.shape

        for scale in self.opt.scales:
            rh, rw = h // (2 ** scale), w // (2 ** scale)
            inputs_equ = {}
            for frame_id in self.opt.frame_ids:
                inputs_equ[frame_id] = []
            
            for batch_idx in range(b):
                src_color = src_colors[batch_idx]
                equ_hist = EqualizeHistTensor(src_color, limit=self._equ_limit)              
                
                for frame_id in self.opt.frame_ids:
                    sci_color = inputs[("color_aug", frame_id, 0)][batch_idx]
                    equ_color = equ_hist(sci_color)
                    if scale != 0:
                        equ_color = F.interpolate(equ_color.unsqueeze(0), (rh, rw), mode='area')
                    if len(equ_color.shape) != 4:
                        equ_color = equ_color.unsqueeze(0)
                    inputs_equ[frame_id].append(equ_color)
                
            for frame_id in self.opt.frame_ids:
                sci_equ = torch.cat(inputs_equ[frame_id])
                if len(sci_equ.shape)== 3:
                    sci_equ = sci_equ.unsqueeze(0)
                inputs[('color_aug', frame_id, scale)] = sci_equ

        return inputs
    

    def get_illu_mask(self, illu):
        illu_mask_high = illu <= 0.95
        illu_mask_low = illu >= 0.35
        illu_mask = illu_mask_high * illu_mask_low
        
        return illu_mask
    
    def auto_illu_mask(self, illu_k):
        illu_k = illu_k.clamp(min=1e-4)
        illu_masks = []
        scales = []
        b, _, _, _ = illu_k.shape
        for i in range(b):
            i_k = illu_k[i]
            # k = 3.141592653589793 / (i_max - i_min)
            # illu_mask = 0.5 * (torch.cos(k * (i_k - i_min)) + 1)
            key_point = self.opt.illu_min * (i_k.median() - i_k.min())
            la = i_k.min() + key_point
            key_point = self.opt.illu_max * (i_k.max() - i_k.median())
            lb = i_k.max() - key_point
            if lb < 0.92:
                lb = 0.92
            else:
                lb = lb
            illu_mask_mid = (i_k > la) * (i_k < lb)
            illu_mask_low = (1 / (1 + (i_k < la) * (i_k - la) * (i_k - la) * (self.opt.p ** 2))) * (i_k < la)
            illu_mask_high = (1 / (1 + (i_k > lb) * (i_k - lb) * (i_k - lb) * (self.opt.q ** 2))) * (i_k > lb)
            illu_mask = illu_mask_mid + illu_mask_low + illu_mask_high
            illu_masks.append(illu_mask)
            k = 1 / (lb - la)
            scales.append(k)
        i_mask = torch.stack(illu_masks)
        k = torch.stack(scales).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return i_mask, k

    
    def get_color_input(self, inputs, frame_id, scale):
        return inputs[("color_aug", frame_id, scale)] if self.opt.use_equ else inputs[("color", frame_id, scale)]

    def generate_images_pred(self, inputs, outputs, scale):
        disp = outputs[("disp", 0, scale)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            T = outputs[("cam_T_cam", 0, frame_id)]
            cam_points = self.backproject(depth, inputs["inv_K", 0])
            pix_coords = self.project_3d(cam_points, inputs["K", 0], T)  # [b,h,w,2]
            src_img = self.get_color_input(inputs, frame_id, 0)
            outputs[("color", frame_id, scale)] = F.grid_sample(src_img, pix_coords, padding_mode="border",
                                                                align_corners=False)
        return outputs

    def compute_reprojection_loss(self, pred, target):
        photometric_loss = robust_l1(pred, target).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
        return reprojection_loss

    def get_static_mask(self, pred, target):
        # compute threshold
        mask_threshold = self.ego_diff.running_val
        # compute diff
        diff = (pred - target).abs().mean(dim=1, keepdim=True)
        # compute mask
        static_mask = (diff > mask_threshold).float()
        # return
        return static_mask
    
    def compute_disp_losses(self, inputs, outputs):
        loss_dict = {}
        if self.opt.use_hist_mask:
            light_color = (inputs[("gray_aug", 0, 0)] * 255).type(torch.uint8)
            light_mask_high = (light_color <= 200)
            light_mask_low = (light_color >= 20)
            light_mask = light_mask_high * light_mask_low
        for scale in self.opt.scales:
            """
            initialization
            """
            disp = outputs[("disp", 0, scale)]
            target = self.get_color_input(inputs, 0, 0)
            reprojection_losses = []

            """
            reconstruction
            """
            outputs = self.generate_images_pred(inputs, outputs, scale)

            """
            automask
            """
            use_static_mask = self.opt.use_static_mask
            use_illu_mask = self.opt.use_illu_mask
            use_hist_mask = self.opt.use_hist_mask

            # update ego diff
            if use_static_mask:
                with torch.no_grad():
                    for frame_id in self.opt.frame_ids[1:]:
                        pred = self.get_color_input(inputs, frame_id, 0)

                        # get diff of two frames
                        diff = (pred - target).abs().mean(dim=1)
                        diff = torch.flatten(diff, 1)

                        # compute quantile
                        quantile = torch.quantile(diff, self.opt.static_mask_quantile, dim=1)
                        mean_quantile = quantile.mean()

                        # update
                        self.ego_diff.update(mean_quantile)

            # compute mask
            for frame_id in self.opt.frame_ids[1:]:
                pred = self.get_color_input(inputs, frame_id, 0)
                color_diff = self.compute_reprojection_loss(pred, target)
                identity_reprojection_loss = color_diff + torch.randn(color_diff.shape).type_as(color_diff) * 1e-5

                # static mask
                if use_static_mask:
                    static_mask = self.get_static_mask(pred, target)
                    identity_reprojection_loss *= static_mask
                    if use_hist_mask:
                        identity_reprojection_loss *= light_mask
                    if use_illu_mask:
                        identity_reprojection_loss *= inputs[("light_mask", 0, 0)]
                        # identity_reprojection_loss *= inputs[("scale_k", 0, 0)]

                reprojection_losses.append(identity_reprojection_loss)

            """
            minimum reconstruction loss
            """
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reproject_loss = self.compute_reprojection_loss(pred, target)
                if use_hist_mask:
                    reproject_loss *= light_mask
                if use_illu_mask:
                    reproject_loss *= inputs[("light_mask", 0, 0)] 
                    # reproject_loss *= inputs[("scale_k", 0, 0)] 
                reprojection_losses.append(reproject_loss)
            reprojection_loss = torch.cat(reprojection_losses, 1)

            min_reconstruct_loss, _ = torch.min(reprojection_loss, dim=1)
            b, _, _ = min_reconstruct_loss.shape
            # max_queue = []
            # for i in range(b):
            #     max_scale = min_reconstruct_loss[i].max()
            #     max_queue.append(max_scale)
            # max_tensor = torch.stack(max_queue)
            # max_tensor = max_tensor.unsqueeze(-1)
            # max_tensor = max_tensor.unsqueeze(-1)
            # min_reconstruct_loss = min_reconstruct_loss / (max_tensor + 1e-4)  
            loss_dict[('min_reconstruct_loss', scale)] = min_reconstruct_loss.mean() / len(self.opt.scales)

            """
            disp mean normalizationmin_reconstruct_loss
            """
            if self.opt.disp_norm:
                mean_disp = disp.mean(2, True).mean(3, True)
                disp = disp / (mean_disp + 1e-7)

            """
            smooth loss
            """
            smooth_loss = get_smooth_loss(disp, self.get_color_input(inputs, 0, scale))
            loss_dict[('smooth_loss', scale)] = self.opt.disparity_smoothness * smooth_loss / (2 ** scale) / len(
                self.opt.scales)

        return loss_dict

class EqualizeHistTensor:
    def __init__(self, tgt_img, limit=0.02):
        """
        Get color map from given RGB image
        :param tgt_img: target image
        :param limit: limit of density
        """
        self._limit = limit
        self._color_map = [self.get_color_map(tgt_img[i, :, :]) for i in range(3)]

    def get_color_map(self, img):
        # get shape
        h, w = img.shape
        num_pixels = h * w
        img = (img * 255)
        # get hist
        flat_img = torch.flatten(img)
        hist = torch.histc(flat_img, bins=256, min=0, max=255)
        limit_pixels = int(num_pixels * self._limit)
        # get number of overflow and clip
        adjust_hist = (hist - limit_pixels).clamp(min=0, max=None)
        num_overflow = adjust_hist.sum()
        hist = hist.clamp(min=0, max=limit_pixels)
        hist += torch.round(num_overflow / 256.0).type(torch.int)
        hist = hist.type(torch.int)
        # get cdf
        cdf = torch.cumsum(hist, dim=0)
        cdf_mask = torch.eq(cdf, 0)
        cdf_m = torch.masked_select(cdf, ~cdf_mask)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min()).type(torch.float32)
        cdf = cdf_m.masked_fill(cdf_mask, 0).type(torch.uint8)
        return cdf

    def __call__(self, img):
        img = (img * 255).type(torch.long)
        chs = [self._color_map[i][img[i, :, :]] for i in range(3)]
        equ_img = torch.stack(chs, axis=0) / 255
        return equ_img
