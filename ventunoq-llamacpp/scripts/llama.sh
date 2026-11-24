#!/bin/bash
# Wrapper: run llama.cpp binaries with bundled libs
export LD_LIBRARY_PATH="$HOME/Code/ventunoq-llamacpp/llama.cpp"
exec "$HOME/Code/ventunoq-llamacpp/llama.cpp/$@"
