# Sugarcane Leaf Disease Dataset - Setup Guide

This guide explains how to use the sugarcane leaf disease dataset with the USB semi-supervised learning framework.

## Dataset Structure

Your dataset should be organized in the following structure:

```
data/
  sugarcane/
    train/
      class1/
        image1.jpg
        image2.jpg
        ...
      class2/
        ...
      class3/
        ...
      class4/
        ...
    test/
      class1/
        ...
      class2/
        ...
      class3/
        ...
      class4/
        ...
```

## Configuration Files

Three config files have been created for testing different SSL algorithms:

1. **FlexMatch**: `config/usb_cv/flexmatch/flexmatch_sugarcane_40_0.yaml`
2. **FreeMatch**: `config/usb_cv/freematch/freematch_sugarcane_40_0.yaml`
3. **SoftMatch**: `config/usb_cv/softmatch/softmatch_sugarcane_40_0.yaml`

### Important Parameters to Adjust

Before running, you may need to adjust these parameters in the config files:

- **num_classes**: Currently set to 4. Update this to match your actual number of classes.
- **num_labels**: Currently set to 40. This is the number of labeled samples to use.
- **img_size**: Currently set to 224. Adjust based on your image sizes.
- **data_dir**: Currently set to `./data`. Update if your data is in a different location.

## Running the Experiments

### 1. FlexMatch

```bash
python train.py --c config/usb_cv/flexmatch/flexmatch_sugarcane_40_0.yaml
```

### 2. FreeMatch

```bash
python train.py --c config/usb_cv/freematch/freematch_sugarcane_40_0.yaml
```

### 3. SoftMatch

```bash
python train.py --c config/usb_cv/softmatch/softmatch_sugarcane_40_0.yaml
```

## Evaluation

After training, evaluate the model:

```bash
python eval.py --dataset sugarcane --num_classes 4 --load_path /path/to/checkpoint
```

## Notes

- The dataset loader automatically splits your training data into labeled and unlabeled sets based on the `num_labels` parameter.
- The labeled/unlabeled split is done in a balanced manner across classes.
- All three algorithms use ResNet18 as the backbone network (configurable via `net` parameter).
- Results will be saved in `./saved_models/usb_cv/` directory.

## Troubleshooting

1. **Dataset not found**: Make sure your dataset is in `./data/sugarcane/` with `train/` and `test/` subdirectories.

2. **Number of classes mismatch**: Update `num_classes` in the config file to match your dataset.

3. **Out of memory**: Reduce `batch_size` in the config file.

4. **Different number of labels**: Create new config files or modify `num_labels` in existing ones.

