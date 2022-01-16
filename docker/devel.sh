#!/bin/bash -e
bash_args=$@
if [[ -z "$bash_args" ]] ; then
    bash_args="bash"
fi

docker run -it \
       -w /workspace \
       -v $(pwd):/workspace \
       kornia_io/devel \
       bash -c "$bash_args"
