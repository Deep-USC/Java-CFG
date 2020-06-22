# Data Preprocessing

Datasets that may be used in this study and the preprocessing scripts of them.

## CFG (Control Flow Graph) Extracting
The [dataset](http://leclair.tech/data/funcom/) (LeClair *et al.* 2019) I used contains functions of Java source code.

### Step 0: Cloning work repository

```bash
wget https://github.com/BohuiZhang/DeepUSC.git
cd DeepUSC/datasets
```

### Step 1: Creating Java resources

```bash
wget https://s3.us-east-2.amazonaws.com/leclair.tech/data/funcom/funcom_filtered.tar.gz
tar -xvzf funcom_filtered.tar.gz
```

### Step 2: Installing extract toolkit 

The CFG extractor I used is [`PROGEX`](https://github.com/ghaffarian/progex). It can be used to extract multiple kinds of graph representation for Java source code. 

```bash
wget https://github.com/ghaffarian/progex/releases/download/v3.4.5/progex-v3.4.5.zip
unzip progex-v3.4.5.zip
```

### Step 3: Running the preprocess script

```bash
chmod +x preprocess.sh
bash preprocess.sh
```

The CFG data in the form of JSON is in the directory `/json` and data in the form of DOT is in the directory `/dot`. One can use `xdot` to visualize DOT files.

## AST (Abstract Syntax Tree) Extracting

