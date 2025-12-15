# Segmentation Fault Fix

## Issues Fixed

I've identified and fixed several issues that were likely causing the segmentation fault:

### 1. **Multiprocessing Configuration** ⚠️ CRITICAL
- **Changed**: `multiprocessing_distributed: True` → `False`
- **Changed**: `num_workers: 4` → `0`
- **Reason**: Multiprocessing with data loading can cause segfaults, especially with PIL image loading. Setting `num_workers: 0` uses the main process for data loading, which is safer for debugging.

### 2. **Resume Flag** 
- **Changed**: `resume: True` → `False`
- **Reason**: If the checkpoint doesn't exist, this can cause issues. Set to `False` for initial training.

### 3. **Error Handling**
- Added error handling in `__sample__` method to catch image loading issues

## Updated Config Files

All three config files have been updated:
- `config/usb_cv/flexmatch/flexmatch_sugarcane_40_0.yaml`
- `config/usb_cv/freematch/freematch_sugarcane_40_0.yaml`
- `config/usb_cv/softmatch/softmatch_sugarcane_40_0.yaml`

## Try Again

Now try running:

```bash
python train.py --c config/usb_cv/flexmatch/flexmatch_sugarcane_40_0.yaml
```

## If It Still Crashes

### Debug Steps:

1. **Check dataset structure**:
   ```bash
   ls -R data/sugarcane/
   ```
   Make sure you have `train/` and `test/` directories with class subdirectories.

2. **Test dataset loading**:
   ```python
   from semilearn.datasets.cv_datasets import get_sugarcane
   from argparse import Namespace
   
   args = Namespace(
       img_size=224,
       crop_ratio=0.875,
       num_labels=40,
       ulb_num_labels=None,
       lb_imb_ratio=1,
       ulb_imb_ratio=1,
       seed=0
   )
   
   lb, ulb, eval_dset = get_sugarcane(
       args, 'flexmatch', 'sugarcane', 40, 4, './data'
   )
   print(f"Labeled: {len(lb)}, Unlabeled: {len(ulb)}, Eval: {len(eval_dset)}")
   ```

3. **Check image files**:
   Make sure all images are valid and can be opened:
   ```python
   from PIL import Image
   import os
   
   for root, dirs, files in os.walk('data/sugarcane/train'):
       for file in files:
           if file.endswith(('.jpg', '.png', '.jpeg')):
               path = os.path.join(root, file)
               try:
                   img = Image.open(path)
                   img.verify()
               except Exception as e:
                   print(f"Bad image: {path} - {e}")
   ```

4. **Reduce batch size**:
   If memory is an issue, try reducing `batch_size` to 4 or 2.

5. **Check CUDA**:
   If using GPU, make sure CUDA is properly installed:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

## After It Works

Once training starts successfully, you can gradually:
- Increase `num_workers` to 2 or 4 (if you have enough CPU cores)
- Set `resume: True` for subsequent runs
- Adjust `batch_size` based on your GPU memory


