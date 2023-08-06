# Do language models have coherent mental models of everyday things?

To investigate this, we propose a benchmark dataset consisting of 100 everyday things, their parts, and the relationships between these parts. We observe that state-of-the-art pre-trained language models (LMs) like GPT-3 and Macaw have fragments of knowledge about these entities, but they fail to produce consistent parts mental models. We propose a simple extension to these LMs where we apply a constraint satisfaction layer on top of raw predictions from LMs to produce more consistent and accurate parts mental models of everyday things. 

These are further described in our paper "Do language models have coherent mental models of everyday things?" (To appear at ACL 2023, Axiv Link: https://arxiv.org/abs/2212.10029)

In this repository, we make the data and code used publicly available.

Note: For files in this repository, “MM” or “mm” is often used as a shorthand for “mental model”.


## Everyday Things Dataset: ParRoT (<u>Par</u>ts and <u>R</u>elations of <u>T</u>hings)

The ParRoT dataset is designed to test whether language models have coherent mental models of everyday things (e.g. egg, tree, flashlight). The dataset contains 11720 relations (p1, rln, p2) across 100 everyday things, 300 mental models, and 2191 parts.

To access the dataset, please refer to https://github.com/allenai/everyday-things/blob/main/data/README_ParRoT_data.md .


## Our proposed approach: ParRoT-Con

Our proposed approach, ParRoT-Con, comprises two main components. The first component “Probing
a Pre-trained Language Model” sends an exhaustive list of relation queries to a LM querying for
every relation between each pair of parts. The second component “constraint reasoning” then applies a constraint satisfaction layer on top of these raw predictions to choose a subset of these relation tuples that are maximally probable and minimally conflicting with each other.


The jupyter notebooks in the code/ folder of this repository provide a step-by-step walkthrough of how we construct the dataset, implement ParRoT-Con for Macaw and GPT3, as well as the analysis performed. The notebooks are intended to be used in the order in which they are labeled. We also provide access to the raw annotations collected in the annotation-files/ folder.


## Setup

### Part1: Creating a conda environent and installing the requirements

We recommend the following steps to setting up an environment for this project:

#### Install conda (if you don't have it)
```
$ wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
$ chmod +x Anaconda*.sh
$ sh Anaconda*.sh
$ export PATH=/home/$USER/anaconda3/bin/:$PATH
```

#### Create a conda environment
The following code snippet creates a new conda environment for this project and installs the requirements inside the conda environment. The requirements.txt file can be found in this repository.

```
$ conda create -n everyday-things python=3.11
$ conda activate everyday-things
```

(This is uncommon, but if your conda environment does not come with pip, you may run the following. Otherwise, skip to the next line and it should start installing the requirements.)
```
$ conda install pip
```

```
$ python3.11 -m pip install -r requirements.txt
```

### Part2: Using your OPENAI API Key

#### Set your ‘OPENAI_API_KEY’ Environment Variable using zsh

The following code snippet is for the Linux / MacOS Set-up, and references https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety. You can also find how to do it for other systems as well as alternative ways there.

1. Run the following command in your terminal, replacing yourkey with your API key
```
$ echo "export OPENAI_API_KEY='yourkey'" >> ~/.zshrc
```

2. Update the shell with the new variable
```
$ source ~/.zshrc
```

3. Confirm that you have set your environment variable using the following command
```
$ echo $OPENAI_API_KEY
```
The value of your API key will be the resulting output.


Now when you open a Jupyter notebook or use python, you’d be able to access your stored key using:
```
os.getenv("OPENAI_API_KEY")
```


# Citation

```
@inproceedings{gu-etal-2023-language,
    title = "Do language models have coherent mental models of everyday things?",
    author = "Gu, Yuling  and
      Dalvi Mishra, Bhavana  and
      Clark, Peter",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.106",
    pages = "1892--1913",
}
```


 
