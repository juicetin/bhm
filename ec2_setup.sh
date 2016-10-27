#!/bin/bash
# ami-3bee4958

sudo pacman -S reflector
sudo reflector --verbose -p http --sort rate --save /etc/pacman.d/mirrorlist

sudo pacman -Syy
sudo pacman -S zsh base-devel git wget tmux unzip zip python-pyqt4
ssh-keygen -t rsa -b 4096 -f $HOME/.ssh/id_rsa -C "justingling@gmail.com" -q -N ""
eval $(ssh-agent -s)
ssh-add
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
localectl set-locale LANG=en_AU.UTF-8
wget https://github.com/jyting/arch-system/raw/master/.zshrc
wget https://github.com/jyting/arch-system/raw/master/.vimrc
wget https://github.com/jyting/arch-system/raw/master/.tmux.conf
wget https://github.com/jyting/arch-system/raw/master/.aliasrc
wget https://github.com/jyting/arch-system/raw/master/.exportrc
wget https://github.com/jyting/arch-system/raw/master/.condarc


wget https://transfer.sh/nT4UX/conda-list &
wget https://transfer.sh/4Fe0x/red-data.zip &
wget https://transfer.sh/xS0Q2/query-sn.npy & 
wget https://transfer.sh/wJhx1/qp-locations.npy & 

chsh root -s /bin/zsh

sudo pacman -Syyu

wget https://transfer.sh/saA3U/anaconda3-4.2.0-linux-x86-64.sh &
bash Anaconda3-4.2.0-Linux-x86_64.sh
conda install --file $HOME/conda-list -n root

wget https://transfer.sh/ml3vM/gp-preds.zip &
wget https://transfer.sh/maWKP/dm-mcmc-chains.zip &
