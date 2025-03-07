from llama_index.llms.bedrock import Bedrock
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic

####################
# This is where we register the LLM clients that can be used by the LLMFactory.
# The LLMFactory will use the registered clients to create the appropriate LLM client.
####################

LLMS = {}


def register_llm_client(llm_client):
    """
    Register a function to create an LLM client.
    """

    def decorator(fn):
        """
        Decorator function to register an LLM client.
        """
        LLMS[llm_client] = fn
        return fn

    return decorator


def get_llm_client(llm_client):
    """
    Retrieve an LLM client by name.
    """
    try:
        return LLMS[llm_client]
    except KeyError:
        raise ValueError(f"LLM client '{llm_client}' is not registered.")


# Register LLM clients
# Add your LLM client registration functions here and they will be automatically registered
# as this module is imported. There is no need to touch the LLMFactory class.


@register_llm_client("openai")
def openai_client(settings):
    """
    Create an OpenAI LLM client.
    """
    print("Creating OpenAI client")
    print(f"Using LLM: {settings.default_model}")
    return OpenAI(
        model=settings.default_model,
        api_key=settings.api_key,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )


@register_llm_client("anthropic")
def anthropic_client(settings):
    """
    Create an Anthropic LLM client.
    """
    print("Creating Anthropic client")
    print(f"Using LLM: {settings.default_model}")
    return Anthropic(
        model=settings.default_model,
        api_key=settings.api_key,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )


@register_llm_client("ollama")
def llama_client(settings):
    """
    Create an Ollama LLM client.
    """
    print("Creating Ollama client")
    print(f"Using LLM: {settings.default_model}")
    return Ollama(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.default_model,
        request_timeout=240.0,
    )


@register_llm_client("bedrock")
def bedrock_client(settings):
    """
    Create a Bedrock LLM client (for Anthropic Claude models)
    """
    print("Creating Bedrock client")
    print(f"Using LLM: {settings.default_model}")
    return Bedrock(
        model=settings.default_model,
        aws_access_key_id=settings.access_key_id,
        aws_secret_access_key=settings.secret_access_key,
        aws_session_token=settings.session_token,
        region_name=settings.default_region,
        context_window=8192,
        request_timeout=120,
    )
