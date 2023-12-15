
# A Universal Unbiased Method for Classification from Aggregate Observations


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training
The code for pairwise similarity/triplet comparison/LLP is in file UUM_stl
And the code for MIL is in file UUM_mil

To train the model(s) in the paper, run this command:

```train
#pairwise  similarity
python main.py -lr 1e-3 -task similarity -batch_size 128 -data_size 120000 -size_m 2 -dataset mnist -so rep -loss rc -ep 100 -init_ep 20 -alpha 0 -cc_init

#triplet comparison
python main.py -lr 1e-3 -task triplet -batch_size 128 -data_size 120000 -size_m 3 -dataset mnist -so rep -loss rc -ep 100 -init_ep 20 -alpha 0 -cc_init

#LLP
python main.py -lr 1e-3 -task llp -batch_size 128 -data_size 120000 -size_m 6 -dataset mnist -so rep -loss rc -ep 100 -init_ep 0 -alpha 0

#MIL
python main.py -lr 2e-1 -batch_size 4096 -dataset musk1 -ep 3500 -init_ep 0 -alpha 1


```