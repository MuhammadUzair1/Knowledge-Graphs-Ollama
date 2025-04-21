FROM ollama/ollama

# Copy custom script
COPY start-ollama.sh /start-ollama.sh
RUN chmod +x /start-ollama.sh

# Override Ollama's entrypoint to allow shell usage
ENTRYPOINT ["/bin/sh"]

# Run your script
CMD ["/start-ollama.sh"]