FROM rust:slim-buster

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    sudo \
    && \
    apt-get clean

RUN python3 -m pip install --upgrade pip setuptools setuptools_rust
