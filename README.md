# PROMO for Personalized Social Emotion Mining (PSEM)
[PROMO](./PRET_SVI.py) algorithm and [eToT](./eToT_new.py) baseline from ECML-PKDD 2020 "PROMO for Interpretable Personalized SocialEmotion Mining". [PROMO](./PRET_SVI.py) uses stochastic variational inference to infer the posterior distributions, while [eToT](./eToT_new.py) uses collapsed Gibbs sampling.

## Data sets:
Facebook Post data: For each post in Facebook, we crawl post content (text), emoticon reactions from users with user ID (hashed). Reading data is through [dataDUE_generator](./dataDUE_generator.py). The data is available at ...
   

## Usage:
Run [experiment.py](./experiment.py) for PROMO and [experiment_eToT.py](./experiment_eToT.py) for eToT.
Packages required: Python2.7, numpy1.16
