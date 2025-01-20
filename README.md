# Playing Hex with Deep Learning

The current solution is ignoring the swap move in the Hex game (second player in the
first move can steal the first player's move, in that case, the color of the piece would
be changed, and it would be moved symmetrically). Current solution also doesn't employ
any search anywhere.

There is no input data (such as games), the network plays against an older version of
itself to get better at the game.

Here is a baseline Deep Learning solver for Hex we are trying to improve upon https://arxiv.org/abs/2104.03113

# Setup

The commands described below works perfectly for Ubuntu systems, as UL IMCS servers are
set up that way.

### Create environment with:
```bash
conda create -n hex python=3.11
conda activate hex
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib
pip install tensorboard
pip uninstall numpy
pip install numpy==1.26.4
```

Note: NumPy installation possibly could be done cleaner.

Note: In the torch installation command `cu181` works for UL IMCS server that Ronalds
uses, but depending on the GPU you or the server has, this string might differ.

### Setting up SOTA solver

To know how good our model is, periodically we want to compare it against the SOTA model
MoHex. To access the MoHex model, we need to set it up from
[this repository](https://github.com/cgao3/benzene-vanilla-cmake).

Although, the SOTA model MoHex has been around for quite a while, the code isn't
perfect, at the moment, as described in
[issue](https://github.com/cgao3/benzene-vanilla-cmake/issues/14), the current master
version isn't working properly, so we need to use version in commit `f88893`.

To set up, use commands like these (the setup path (first line) might differ, but that's
fine, you will just have to update it in the code):
```bash
cd /tmp
git clone https://github.com/cgao3/benzene-vanilla-cmake.git
cd benzene-vanilla-cmake/
git reset --hard f88893
mkdir build
cd build
sudo apt-get install libboost-all-dev
sudo apt-get install libdb-dev
cmake ../
make -j4
```

To test out the setup, additionally to the commands above run:
```bash
cd ..
./build/src/mohex/mohex
```

Then you can use commands, for example, like:
```
list_commands
showboard
genmove black
play white f6
boardsize 11 11
...
```

# Usage

### Run longer training sessions with:
```bash
nohup python -u main.py >> output.txt &
```

### See logged results with:
```bash
tensorboard --logdir="runs"
```

### See logged server results with:
```bash
ssh root@<server_ip> -p 7001 -L 6006:localhost:7002
<enter_the_password>
cd /tmp/deep_hex
conda activate hex
tensorboard --logdir="runs" --port=7002
```
