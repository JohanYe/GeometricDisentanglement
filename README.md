# GeometricDisentanglement

Source code for my Master's thesis at the Technical University of Denmark 2021.

Investigating disentanglement using a non-euclidean approach.

Training variational autoencoder example:
```
python train_vae.py --model_dir  vae --dataset bodies 
```

Training LAND to latent representations:
```
python fit_land.py --model_dir  vae --exp_name experiment_1 --dataset bodies --hpc
```

Cleaner code and repo to come
