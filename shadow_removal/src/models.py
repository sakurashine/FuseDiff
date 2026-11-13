import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lpips
from .networks import InpaintGenerator
from .diffusion import DiffusionRefiner



class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.model_save = config.PATH  


    def load(self, model_path):
        # resolve relative path against checkpoint directory
        if not os.path.isabs(model_path):
            model_path = os.path.join(self.model_save, model_path)

        self.gen_weights_path = model_path

        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator from %s' % (self.name, self.gen_weights_path))

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            try:
                self.generator.load_state_dict(data['generator'])
            except RuntimeError as e:
                print(f'Warning: load_state_dict with non-strict mode due to: {e}')
                self.generator.load_state_dict(data['generator'], strict=False)
            if 'diffusion' in data and hasattr(self, 'diffusion'):
                print('Loading diffusion refiner from checkpoint')
                self.diffusion.load_state_dict(data['diffusion'])
            self.iteration = data['iteration']
        else:
            print('Warning: weights not found at %s, using random init.' % self.gen_weights_path)

    def save(self, extend=''):

        if len(self.config.GPU) > 1:
            generate_param = self.generator.module.state_dict()
            print('save...multiple GPU')
        else:
            generate_param = self.generator.state_dict()
            print('save...single GPU')

        save_dict = {
            'iteration': self.iteration,
            'generator': generate_param
        }
        if hasattr(self, 'diffusion'):
            save_dict['diffusion'] = self.diffusion.state_dict()

        torch.save(save_dict, os.path.join(self.model_save, '{}_{}_gen_{}.pth'.format(self.iteration, self.name, extend)))

        print('\nsaving %s...\n' % self.name)


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        generator = InpaintGenerator(config=config)
        self.add_module('generator', generator)

        self.freeze_coarse = bool(getattr(config, 'DIFF_FREEZE_COARSE', 1))
        if self.freeze_coarse:
            for p in generator.parameters():
                p.requires_grad = False

        self.diffusion_enabled = bool(getattr(config, 'DIFF_ENABLE', True))
        if self.diffusion_enabled:
            diffusion = DiffusionRefiner(config=config)
            self.add_module('diffusion', diffusion)
        else:
            diffusion = None

        self.add_module('l1_loss', nn.L1Loss())
        self.lpips = lpips.LPIPS(net='vgg', verbose=False).to(config.DEVICE) if getattr(config, 'DIFF_LPIPS_WEIGHT', 0) > 0 else None

        # optimizers
        self.gen_optimizer = None if self.freeze_coarse else optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.diff_optimizer = None
        if diffusion is not None:
            self.diff_optimizer = optim.AdamW(
                params=diffusion.parameters(),
                lr=float(getattr(config, 'DIFF_LR', config.LR)),
                betas=(config.BETA1, config.BETA2)
            )

    def calculate_loss(self, coarse, refined, images, masks, diff_losses=None):

        gen_l1_loss = self.l1_loss(coarse, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        total_loss = gen_l1_loss * float(getattr(self.config, 'COARSE_L1_WEIGHT', 1.0))
        logs = [("l_coarse_l1", gen_l1_loss.item())]

        if self.diffusion_enabled and diff_losses is not None:
            diff_l2 = diff_losses.get('l2')
            if diff_l2 is not None:
                weight = float(getattr(self.config, 'DIFF_L2_WEIGHT', 1.0))
                total_loss = total_loss + weight * diff_l2
                logs.append(("l_diff_l2", diff_l2.item()))

            diff_lpips = diff_losses.get('lpips') if isinstance(diff_losses, dict) else None
            if diff_lpips is not None:
                weight = float(getattr(self.config, 'DIFF_LPIPS_WEIGHT', 0.0))
                total_loss = total_loss + weight * diff_lpips
                logs.append(("l_diff_lpips", diff_lpips.item()))

        return total_loss, logs

    def process(self, images, masks, shadows):
        self.iteration += 1

        if self.gen_optimizer:
            self.gen_optimizer.zero_grad()
        if self.diff_optimizer:
            self.diff_optimizer.zero_grad()

        time_steps = None
        diff_time_steps = None
        if getattr(self.config, 'AFN_TIME_COND_ENABLE', False):
            max_t = max(int(getattr(self.config, 'AFN_TIME_STEPS', 1000)), 1)
            if self.training:
                time_steps = torch.randint(1, max_t + 1, (shadows.size(0),), device=shadows.device, dtype=torch.long)
                diff_T = max(int(getattr(self.config, 'DIFF_TRAIN_STEPS', max_t)), 1)
                diff_time_steps = torch.clamp(
                    torch.round(time_steps.float() / max_t * diff_T), 1, diff_T
                ).long()
            else:
                infer_t = max(int(getattr(self.config, 'AFN_INFER_T', 0)), 0)
                infer_t = min(infer_t, max_t)
                time_steps = torch.full((shadows.size(0),), infer_t, device=shadows.device, dtype=torch.long)

        coarse = self(shadows, masks, time_steps=time_steps)

        refined = coarse
        diff_losses = {}
        if self.diffusion_enabled and self.training:
            pred_x0, _ = self.diffusion.forward_train(
                images=images,
                shadows=shadows,
                masks=masks,
                coarse=coarse,
                t_override=diff_time_steps,
            )
            refined = pred_x0
            diff_losses['l2'] = F.mse_loss(pred_x0, images)
            if self.lpips is not None:
                diff_losses['lpips'] = self.lpips(pred_x0 * 2 - 1, images * 2 - 1).mean()
        elif self.diffusion_enabled and (not self.training):
            refined = self.diffusion.forward_sample(shadows=shadows, masks=masks, coarse=coarse)

        gen_loss, logs = self.calculate_loss(coarse=coarse, refined=refined, images=images, masks=masks, diff_losses=diff_losses)

        return coarse, refined, gen_loss, logs

    def forward(self, shadows, masks, time_steps=None):
        shadows_masked = shadows * (1 - masks)

        inputs_up = torch.cat((shadows_masked, masks), dim=1)
        inputs_down = torch.cat((shadows, masks), dim=1)

        outputs_down = self.generator(inputs_down, inputs_up, time_steps=time_steps)
        return outputs_down

    def backward(self, gen_loss=None):
        gen_loss.backward()
        if self.gen_optimizer:
            self.gen_optimizer.step()
        if self.diff_optimizer:
            self.diff_optimizer.step()

