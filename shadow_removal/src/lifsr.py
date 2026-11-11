import os
import lpips
import numpy as np
import torch
from skimage.color import rgb2lab
from torch.utils.data import DataLoader
from thop import profile
from .dataset import Dataset
from .metrics import PSNR
from .models import InpaintingModel
from .utils import save_sample_png
import time

class LIFSR():
    def __init__(self, config):
        self.config = config

        self.debug = False
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)


        self.psnr = PSNR(255.0).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_GT_FLIST, config.TEST_MASK_FLIST, config.TEST_MASK2_FLIST, config.TEST_SHADOW_FLIST, augment=False, training=False)
            print('test dataset:{}'.format(len(self.test_dataset)))
        else:
            self.train_dataset = Dataset(config, config.TRAIN_GT_FLIST, config.TRAIN_MASK_FLIST, config.TRAIN_MASK_FLIST, config.TRAIN_SHADOW_FLIST, augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_GT_FLIST, config.VAL_MASK_FLIST, config.VAL_MASK2_FLIST, config.VAL_SHADOW_FLIST, augment=False, training=False)

            print('train dataset:{}'.format(len(self.train_dataset)))
            print('eval dataset:{}'.format(len(self.val_dataset)))

    def load(self):
        self.inpaint_model.load(self.config.MODEL_LOAD)

    def save(self, extend=''):
        self.inpaint_model.save(extend)

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=16,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        min_mae_all = 100
        min_mae_shadow = 100
        min_mae_non = 100
        while(keep_training):
            epoch += 1  
            print('\n\nTraining epoch: %d' % epoch)

            for items in train_loader:
                self.inpaint_model.train()

                images, masks, masks2, shadows = self.cuda(*items)

                # train
                outputs_coarse, outputs_refined, gen_loss, logs = self.inpaint_model.process(images, masks, shadows)

                # backward
                self.inpaint_model.backward(gen_loss)
                iteration = self.inpaint_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs


                # sample
                if iteration % self.config.TRAIN_SAMPLE_INTERVAL == 0:
                    img_list2 = [shadows * (1 - masks), images, outputs_coarse, outputs_refined]
                    name_list2 = ['in', 'gt', 'coarse', 'refine']
                    save_sample_png(sample_folder=self.config.TRAIN_SAMPLE_SAVE,
                                              sample_name='ite_{}_{}'.format(self.inpaint_model.iteration,
                                                                             0), img_list=img_list2,
                                              name_list=name_list2, pixel_max_cnt=255, height=-1,
                                              width=-1)


                # save model at checkpoints
                if iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

                # evaluate model at checkpoints
                if iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    mae_all, mae_shadow, mae_non = self.eval()
                    self.inpaint_model.iteration = iteration

                    save = False
                    if mae_all < min_mae_all:
                        min_mae_all = mae_all
                        save = True
                    if mae_shadow < min_mae_shadow:
                        min_mae_shadow = mae_shadow
                        save = True
                    if mae_non < min_mae_non:
                        min_mae_non = mae_non
                        save = True

                    if save:
                        self.save(extend=f'{round(min_mae_all, 4)}_{round(min_mae_shadow, 4)}_{round(min_mae_non, 4)}')
                        print('---mae decrease-iteration:{}'.format(iteration))

                print(logs)

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            num_workers=8,
            drop_last=True,
            shuffle=False
        )


        self.inpaint_model.eval()

        mae_list = []

        shadow_list = []
        non_list = []

        iteration = self.inpaint_model.iteration
        with torch.no_grad():
            for items in val_loader:
                images, masks, masks2, shadows = self.cuda(*items)

                # eval
                outputs_coarse, outputs_refined, gen_loss, logs = self.inpaint_model.process(images, masks, shadows)

                mae_all = self.metric(images, outputs_refined, masks2, shadow_list, non_list)
                mae_list.append(mae_all)

                # sample
                if len(mae_list) % self.config.EVAL_SAMPLE_INTERVAL == 0:
                    img_list2 = [shadows * (1 - masks), images, outputs_coarse, outputs_refined]
                    name_list2 = ['in', 'gt', 'coarse', 'refine']
                    save_sample_png(sample_folder=self.config.EVAL_SAMPLE_SAVE,
                                          sample_name='ite_{}_{}'.format(iteration, len(mae_list)), img_list=img_list2,
                                          name_list=name_list2, pixel_max_cnt=255, height=-1,
                                          width=-1)



                print(f"mae_all:{round(mae_all, 4)}/{round(np.average(mae_list), 4)}  {len(mae_list)}")

            shadow_np = np.array(shadow_list)
            non_np = np.array(non_list)
            shadow_area = np.sum(shadow_np[:, 0]) / np.sum(shadow_np[:, 1])
            non_area = np.sum(non_np[:, 0]) / np.sum(non_np[:, 1])

            print('iteration:{} ave_mae: all:{} shadow:{}  non:{}'.format(iteration, round(np.average(mae_list), 4),
                                                                          round(np.average(shadow_area), 4),
                                                                          round(np.average(non_area), 4)))

            return np.average(mae_list), shadow_area, non_area

    def test(self):
        self.inpaint_model.eval()

        device = torch.device(self.config.DEVICE)

        dummy_images = torch.randn(1, 3, 1024, 1024, device=device)
        dummy_masks = torch.zeros(1, 1, 1024, 1024, device=device)
        dummy_shadows = torch.randn(1, 3, 1024, 1024, device=device)

        class InferenceWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, images, masks, shadows):
                _, outputs, _, _ = self.model.process(images, masks, shadows)
                return outputs

        wrapper = InferenceWrapper(self.inpaint_model).to(device)
        wrapper.eval()

        with torch.no_grad():
            macs, params = profile(
                wrapper,
                inputs=(dummy_images, dummy_masks, dummy_shadows),
                verbose=False
            )

        params_m = params / 1e6
        gmacs = macs / 1e9

        gflops = 2 * gmacs

        print(f"Params: {params_m:.2f} M")
        print(f"MACs: {gmacs:.2f} GMACs")
        print(f"FLOPs: {gflops:.2f} GFLOPs")

        os.makedirs(self.config.TEST_SAMPLE_SAVE, exist_ok=True)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        mae_list = []

        shadow_list = []
        non_list = []

        total_inference_time = 0.0  
        warmup_done = False  

        with torch.no_grad():
            peak_memory_list = []  
            for items in test_loader:
                images, masks, masks2, shadows = self.cuda(*items)

                if not warmup_done:
                    print("GPU warming up...")
                    for _ in range(10):
                        _ = self.inpaint_model.process(images, masks, shadows)
                    torch.cuda.synchronize(images.device)
                    warmup_done = True
                    print("GPU warm-up completed.")

                torch.cuda.synchronize(images.device)  
                torch.cuda.reset_peak_memory_stats(images.device)  
                
                test_start = time.time()
                _, outputs, _, _ = self.inpaint_model.process(images, masks, shadows)
                torch.cuda.synchronize(images.device)  
                test_end = time.time()
                peak_memory = torch.cuda.max_memory_allocated(images.device) / (1024 ** 2)
                peak_memory_list.append(peak_memory)
                
                inference_time = test_end - test_start
                total_inference_time += inference_time

                mae_all = self.metric(images, outputs, masks2, shadow_list, non_list)
                mae_list.append(mae_all)

                print(f"mae_all:{round(mae_all, 4)}/{round(np.average(mae_list), 4)}  {len(mae_list)}")

                if len(mae_list) % 1 == 0:
                    base_name = self.test_dataset.name.split('.')[0]
                    save_sample_png(sample_folder=self.config.TEST_SAMPLE_SAVE,
                                    sample_name=base_name,
                                    img_list=[outputs],
                                    name_list=[''], pixel_max_cnt=255, height=-1, width=-1)

            shadow_np = np.array(shadow_list)
            non_np = np.array(non_list)
            shadow_area = np.sum(shadow_np[:, 0]) / np.sum(shadow_np[:, 1])
            non_area = np.sum(non_np[:, 0]) / np.sum(non_np[:, 1])
            print('iteration: {}  mae: all:{} shadow:{} non:{}'.format(self.inpaint_model.iteration,
                                                                       round(np.average(mae_list), 4),
                                                                       round(shadow_area, 4), round(non_area, 4)))
            
            print(f"Total inference time (forward only): {total_inference_time:.2f} seconds")
            print(f"Average inference time: {total_inference_time / len(mae_list) * 1000:.2f} ms/image")
            print(f"FPS: {len(mae_list) / total_inference_time:.2f}")   
            peak_memory = max(peak_memory_list)
            print(f"Peak Inference GPU Memory: {peak_memory:.2f} MiB")
            with open(os.path.join(self.config.TEST_SAMPLE_SAVE, "inference_time.txt"), "a", encoding="utf-8") as f:
                f.write(f"Total inference time (forward only): {total_inference_time:.4f} seconds\n"
                        f"Average inference time: {total_inference_time / len(mae_list) * 1000:.2f} ms/image\n"
                        f"FPS: {len(mae_list) / total_inference_time:.2f}\n"
                        f"Peak GPU Memory: {peak_memory:.2f} MiB\n"
                        f"Params: {params_m:.2f} M\n"
                        f"MACs (1024x1024): {gmacs:.2f} GMACs\n"
                        f"FLOPs (1024x1024): {gflops:.2f} GFLOPs\n"
                        f"{'-' * 40}\n")

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def metric(self, gt, pre, mask, shadow_list, non_list):
        pre = pre.clamp_(0, 1) * 255.0
        pre = pre.permute(0, 2, 3, 1)
        pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

        gt = gt.clamp_(0, 1) * 255.0
        gt = gt.permute(0, 2, 3, 1)
        gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

        mask = mask.cpu().data.numpy()[0][0].astype(np.uint8)

        f = gt / 255
        s = pre / 255

        f = rgb2lab(f)
        s = rgb2lab(s)

        absl = abs(f[:, :, 0] - s[:, :, 0])
        absa = abs(f[:, :, 1] - s[:, :, 1])
        absb = abs(f[:, :, 2] - s[:, :, 2])

        smask = np.ones([1024, 1024])
        summask = np.sum(smask)

        l = np.sum(absl) / summask
        a = np.sum(absa) / summask
        b = np.sum(absb) / summask

        result_all = l + a + b

        # ------------------------------shadow
        l_shadow = np.sum(absl * mask)
        a_shadow = np.sum(absa * mask)
        b_shadow = np.sum(absb * mask)

        result_shadow = l_shadow + a_shadow + b_shadow
        shadow_list.append([np.sum(result_shadow), np.sum(mask)])

        # ------------------------------non shadow
        l_non = np.sum(absl * (1 - mask))
        a_non = np.sum(absa * (1 - mask))
        b_non = np.sum(absb * (1 - mask))

        result_non = l_non + a_non + b_non
        non_list.append([np.sum(result_non), np.sum(1 - mask)])

        return result_all