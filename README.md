
# A Universal Unbiased Method for Classification from Aggregate Observations



## Training

To train the model(s) in the paper, run this command:

```train
#pairwise  similarity
python main.py -lr 1e-3 -task similarity -size_m 2 -dataset mnist -so rep -loss rc 

#triplet comparison
python main.py -lr 1e-3 -task triplet -size_m 3 -dataset mnist -so rep -loss rc


```
