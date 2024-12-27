from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.chat_models.huggingface import ChatHuggingFace
from logging import getLogger

from src.config import LLMConf

logger = getLogger(__name__)


def fetch_llm(conf: LLMConf) -> BaseChatModel | None:
    """
    Fetches the LLM model.
    """
    logger.info(f"Fetching LLM model '{conf.model}'..")

    if conf.type == "ollama":
        llm = ChatOllama(
            model=conf.model,
            temperature=conf.temperature
        )
    elif conf.type == "openai":
        llm = ChatOpenAI(
            model=conf.model,
            api_key=conf.api_key,
            deployment=conf.deployment,
            temperature=conf.temperature,
        )
    elif conf.type == "trf":
        llm = ChatHuggingFace(
            model=conf.model,
            endpoint=conf.endpoint,
            temperature=conf.temperature,
        )

    else:
        logger.warning(f"LLM type '{conf.type}' not supported.")
        llm = None
    
    logger.info(f"Initialized LLM of type: '{conf.type}'")
    return llm 