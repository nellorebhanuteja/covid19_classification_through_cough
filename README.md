<br /><br />
## 1. About:

This is a system written with a python back-end and a shell front-end, and follows object oriented programming. This is originally developed for classifying COVID19 coughs from non-COVID19 coughs. It is composed of the following parts:
- Feature extraction
- Model initialization
- Model training, and exporting
- Validation data classification
- Performance computation

<br /><br />
## 2. Directory structure:

.
├── LICENSE.md
├── Readme
├── conf
│   ├── feature.conf
│   ├── train.conf
│   ├── train_lr.conf
│   ├── train_mlp.conf
│   └── train_rf.conf
├── feature_extraction.py
├── infer.py
├── models.py
├── parse_options.sh
├── REQUIREMENTS.txt
├── run.sh
├── scoring.py
├── summarize.py
└── train.py

<br /><br />
## 3. Directory contents:


- conf/
	-- feature.conf				[ Configuration file used by the feature extraction module ]
    -- train_lr.conf            [ Configuration file used by LR classifier module ]
    -- train_mlp.conf           [ Configuration file used by MLP classifer module ]
    -- train_rf.conf            [ Configuration file used by RF classifer module ]

- run.sh					    [ Master (shell) script to run the codes ]
- parse_options.sh				[ Facilitates inputing command-line arguments to run.sh (borrowed
                                from Kaldi, note the license details inside it)]

- feature_extraction.py         [ Extract features: requires feature configuration, and the list
                                of files wav.scp ]	

- models.py                     [ Model definition: contains models details]
- train.py                      [ Training models: uses models.py ]
- infer.py                      [ Inference: forward pass through the trained model to generate
                                score as probalities]
- scoring.py                    [ Performance: computes false positives, true positives, etc.,
                                from ground truth labels and the system scores ]
- summarize.py                  [ Summarize: document the results across the folds and generate
                                average performance metrics ]
- REQUIREMENTS.txt              [ Contains a list of dependencies to run the system ]





<br /><br />
## 4. How to create database directory:

Directory structure
```
│   README.md  
│   LICENSE.md    
│   METADATA.csv  
└───AUDIO
│   │   <file_name>.wav
└───LISTS
    │   train_fold_<fold_num>.txt
    │   val_fold_<fold_num>.txt
```

Directory contents
```
	AUDIO: Contains audio files
	LISTS: Contains train and validation lists for 5 folds
    metadata.csv: Contains the subject information for each every audio file
```

Audio file description
```
Audio file can be of any format such as flac, wav etc..
```
metadata.csv header description
```
	File_name   : filename (without extension)
	Covid_status: COVID+ve (p) / Non-COVID (n)
	Gender      : Male (m) / Female (f)
	Nationality : India (I) / Other (O)
```


<br /><br />
## 4. How to run the script:

- Install anaconda in your system

- Create environment
    ```
    conda env create -f environment.yml
    ```
- Activate the environment
    ```
    conda activate INDICON2021
    ```
- The configuration used in the paper can be found at 
    ```
        conf/feature.conf
    ```
    (Change the various values present in conf folder for experimenting.)

- Open run.sh file and assign values for the following variables
    - stage: the run.sh file is divided into stages. This variable defines until which stage the file runs.
    - listdir: location of LISTS folder
    - audiodir:location of AUDIO folder
    - metadatafile: location of metadata.csv
    - datadir: location where certain text files required for training are saved
    - feature_dir: location where the generated features are saved
    - output_dir: location where outuputs (including predictions, scores and plots) are saved
    - train_config: location of train.conf
    feats_config: location of feature.conf

-   ```
    bash run.sh
    ```

