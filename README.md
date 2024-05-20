## About
**NotYetAnotherNightshade** (NYAN) is a graph variational encoder as described in the article "Application of variational graph encoders as an effective generalist algorithm in holistic computer-aided drug design" published in Nature Machine Intelligence, 2023. In NYAN, the low-dimension latent variables derived from the variational graph autoencoder are leveraged as a kind of universal molecular representation, yielding remarkable performance and versatility throughout the drug discovery process.

We assess the reusability of NYAN and comprehensively investigate its applicability within the context of specific chemical toxicity prediction. We used more expanded predictive toxicology datasets sourced from TOXRIC, a comprehensive and standardized toxicology database (<span style="color:red;">Lianlian Wu, Bowei Yan, Junshan Han, Ruijiang Li, Jian Xiao, Song He, Xiaochen Bo. TOXRIC: a comprehensive database of toxicological data and benchmarks, Nucleic Acids Research, 2022, https://toxric.bioinforai.tech/home</span>).

This repository contains the code we used to perform the multi-task learning experiments on acute toxicity dataset of TOXRIC. We firstly used the re-trained NYAN 325K model (see our another repository `NYAN_reuse`: https://github.com/LuJiangTHU/NYAN_reuse) to derive five kinds of random NYAN latent representations,  for all the 80,081 chemical compounds in acute toxicity dataset, which are saved into `NYAN_latent0.txt`, `NYAN_latent1.txt`, `NYAN_latent2.txt`, `NYAN_latent3.txt`, and `NYAN_latent4.txt`, respectively. These 5 files should be place in the path `./data/`. In addition, there also should be a molecular feature dataset file `all_descriptors.txt` in the in the path `./data/`, and Avalon features are contained by this dataset file. 

However, the above 6 dataset files are all too large to upload to GitHub. If you need them, please contact me directly by email `lu-j13@tsinghua.org.cn`.


## Code environment
```
python==3.11.5
torch==2.2.0
torchaudio==2.2.0
torchnet==0.0.4
pandas==2.1.4
```

## Training your MT-NYAN
The `./config/cfg_Avalon+NYAN_fold0_lat0.py` can be used to control the training configurations and model architecture. Using the following command to train your MT-NYAN model:
```sh
python train.py --config cfg_Avalon+NYAN_fold0_lat0 
```
The optimal model will be saved into `./experiments/cfg_Avalon+NYAN_fold0_lat0` correspondingly.

## Evaluating 
After trained using 5 cross-validation folds and 5 NYAN latent dataset files, you can use the `MTL_consensus_evaluation.py` to evaluate the final averaged  performance on 5 cross-validation folds:
```sh
python MTL_consensus_evaluation.py
```
The results will be saved as the form of tables and placed under the folder  `./table_results/`. 



