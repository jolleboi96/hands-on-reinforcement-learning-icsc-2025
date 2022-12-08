# Hands-on Reinforcement Learning excercise
## Optimizing an RF double splitting using Stable Baselines 3

In this gitlab repository you will find the code used for the special SY-RF-BR Data analysis and machine learning meeting held on the 15/12/2022.

# Instructions

This excercise is recommended to be run locally on your own python distribution in a new virtual environment. You can run it in SWAN, but the training of agents tends to be very slow.

## Clone/download the Gitlab repository to your local computer

You can do this either through git clone or simply downloading the .zip file of the entire repository and then unpacking it somewhere on your local drive (or somewhere else you can access it, like eos).

## Creating a new python virtual environment and install the project dependencies

It is assumed that you have a working python distribution installed on your system. Then, from a terminal you can run the command

<code>python3 -m venv /path/to/new/virtual/environment </code> if you are running a pure python distribution, or
<code>conda create --name myenv </code>, if you are using anaconda.

After your virtual environment is created, you also need to activate it in your terminal. This is done through the following command:

<code>python3 source /path/to/new/virtual/environment/bin/activate </code> if you are running a pure python distribution, or
<code>conda activate myenv </code>, if you are using anaconda.

### Installing required packages in your venv

With your venv activated, you can now install the necessary packages for the notebook excercise. This is done by directing pip to a requirements.txt file in this gitlab repository, which you have downloaded already. In your terminal, with your new virtual environment activated, navigate to the local folder of your donwloaded project files and run the following command,

<code>pip install -r requirements.txt</code>

This should install all packages you will need to run the notebook.
