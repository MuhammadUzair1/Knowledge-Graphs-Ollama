#!/bin/sh

ollama pull llama3.2
ollama pull mxbai-embed-large

exec ollama serve