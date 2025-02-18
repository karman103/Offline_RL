# ViZDoom-Two-Colors

## Dataset

First, create the `VizDoom/VizDoom_data/iterative_data/` directory.
Then a dataset can be generated by executing the `VizDoom/VizDoom/VizDoom_notebooks/generate_iter_data.ipynb` file.

## Example usage

```python
python3 VizDoom/VizDoom_src/train_vizdoom.py --model_mode 'RATE' --arch_mode 'TrXL' --ckpt_folder 'RATE_ckpt' --text 'my_comment'
```

Where:

1. `model_mode` - select model:
    - 'RATE' -- our model
    - 'DT' -- Decision Transformer model
    - 'DTXL' -- Decision Transformer woth caching hidden states
    - 'RATEM' -- RATE without caching hidden states
    - 'RATE_wo_nmt' -- RATE without memory embeddings
2. `arch_mode` - select backbone model:
    - 'TrXL'
    - 'TrXL-I'
    - 'GTrXL'

