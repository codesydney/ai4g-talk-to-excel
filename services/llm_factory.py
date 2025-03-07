from config.app_settings import get_settings
import services.llm_registrations as llm_registrations


class LLMFactory:
    """
    Factory class to create LlamaIndex client using the specified LLM provider.
    """

    def __init__(self, provider: str):
        """
        Initialize the LLMFactory with the specified provider.
        """
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        self.client = llm_registrations.get_llm_client(self.provider)(self.settings)
