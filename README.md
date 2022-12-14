# Hands-on Reinforcement Learning excercise
## Optimizing an RF double splitting using Stable Baselines 3

In this gitlab repository you will find the code used for the special SY-RF-BR Data analysis and machine learning meeting held on the 15/12/2022.

# Instructions

This excercise is recommended to be run locally on your own python distribution in a new virtual environment. You can run it in SWAN, but the training of agents tends to be very slow.

## Clone/download the Gitlab repository to your local computer

You can do this either through git clone or simply downloading the .zip file of the entire repository and then unpacking it somewhere on your local drive (or somewhere else you can access it, like eos).

## Creating a new python virtual environment and install the project dependencies

It is assumed that you have a working python distribution installed on your system. Depending on whether you are using a basic python distribution or Anaconda, check the relevation sections below.

### Using Conda

From an anaconda prompt terminal, navigate to the folder in which you have placed the files of this repository. Then, run the command to create a new virtual environment with the required dependencies for this excercise.

<code>conda create --name rl-hands-on</code>

The name of the new environment will be <code>rl-hands-on</code>.

After your virtual environment is created, you also need to activate it in your terminal. This is done through the following command:

<code>conda activate rl-hands-on </code>, if you are using anaconda.

#### Installing packages (conda)

Run the following commands to install the necessary packages.

<code>conda install pytorch torchvision cpuonly -c pytorch</code>

<code>pip install gym matplotlib numpy scipy jupyterlab stable-baselines3 tensorboard PyQt5</code>

### Using pip (standard python)
From a terminal, write the command
<code>python3 -m venv /path/to/new/virtual/environment </code> to create your venv.
After your virtual environment is created, you also need to activate it in your terminal. This is done through the following command:
<code>source /path/to/new/virtual/environment/bin/activate </code>

#### Installing required packages in your venv

With your venv activated, you can now install the necessary packages for the notebook excercise. This is done by directing pip to a requirements.txt file in this gitlab repository, which you have downloaded already. In your terminal, with your new virtual environment activated, navigate to the local folder of your donwloaded project files and run the following command,

<code>pip install -r requirements.txt</code>

This should install all packages you will need to run the notebook.

If there are any issues using the requirements.txt file to install your packages, the ones you need are as follows:

<pre><code> 
pip install stable-baselines3 numpy matplotlib torchvision gym scipy jupyterlab torch PyQt5
</pre></code> 
