tmux\
    new-session -d -s marekvasko \; \
    send "/usr/bin/singularity shell --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:22.01-py3.SIF" ENTER ;
#ANY COMMAND GOES HERE
    split-window -h 'htop -u marekvasko' \; \
    split-window -v \; \
    attach-session

