"""Bard chat wrapper."""
import inspect
import logging
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from langchain.callbacks.manager import CallbackManager, Callbacks
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM
from langchain.schema import (
    Generation, 
    ChatGeneration, 
    ChatMessage, 
    ChatResult,
    AIMessage,
    BaseMessage,
    SystemMessage,
    HumanMessage, 
    LLMResult, 
    get_buffer_string)
from langchain.utils import get_from_dict_or_env
from pydantic import Extra, Field, root_validator
from tenacity import (before_sleep_log, retry, stop_after_attempt,
                      wait_exponential)

logger = logging.getLogger(__name__)

class BardChat(BaseChatModel):
    """Wrapper around Bard large language models.

    To use, you should have the ``GoogleBard``[https://github.com/acheong08/Bard] python package installed, and the
    environment variable ``BARD_SESSION_ID`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.
    """

    client: Any  #: :meta private:
    model_name: str = "Bard"
    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    bard_session_id: Optional[str] = None
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to OpenAI completion API. Default is 600 seconds."""
    max_retries: int = 3
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    prompt_tokens: Optional[int] = None
    """Prompt mesagge token"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        disallowed_model_kwargs = all_required_field_names | {"model"}
        invalid_model_kwargs = disallowed_model_kwargs.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        bard_session_id = get_from_dict_or_env(
            values, "bard_session_id", "BARD_SESSION_ID"
        )
        try:
            from Bard import Chatbot

        except ImportError:
            raise ValueError(
                "Could not import GoogleBard python package. "
                "Please install it with `pip install GoogleBard`."
            )
        try:
            values["client"] = Chatbot(bard_session_id)
        except AttributeError:
            raise ValueError(
                "you can't use this BARD_SESSION_ID"
                "Go to https://bard.google.com/"
                "F12 for console"
                "Session: Go to Application → Cookies → __Secure-1PSID. Copy the value of that cookie."
                "Set your env BARD_SESSION_ID = {copied value}"
            )
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "model": self.model_name,
            "request_timeout": self.request_timeout,
            "max_tokens": self.max_tokens,
            "n": self.n,
            **self.model_kwargs,
        }

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            for k, v in token_usage.items():
                if k in overall_token_usage:
                    overall_token_usage[k] += v
                else:
                    overall_token_usage[k] = v
        return {"token_usage": overall_token_usage, "model_name": self.model_name}

    def generate(
        self,
        messages: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """Top Level call"""

        callback_manager = CallbackManager.configure(
            callbacks, self.callbacks, self.verbose
        )
        if isinstance(messages[0], str):
            messages = [[HumanMessage(content = m)] for m in messages]
        message_strings = [get_buffer_string(m) for m in messages]
        run_manager = callback_manager.on_llm_start(
            {"name": self.__class__.__name__}, message_strings
        )

        new_arg_supported = inspect.signature(self._generate).parameters.get(
            "run_manager"
        )
        try:
            results = [
                self._generate(m, stop=stop, run_manager=run_manager)
                if new_arg_supported
                else self._generate(m, stop=stop)
                for m in messages
            ]
        except (KeyboardInterrupt, Exception) as e:
            run_manager.on_llm_error(e)
            raise e
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        output = LLMResult(generations=generations, llm_output=llm_output)
        run_manager.on_llm_end(output)
        return output

    def _generate(
        self,
        messages: str,
        stop: Optional[List[str]] = None,
    ) -> ChatResult:
        self.prompt_tokens = self.get_num_tokens(messages)
        params = self._create_message_dicts(stop)
        response = completion_with_retry(self, messages=messages, **params)
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: str,
        stop: Optional[List[str]] = None,
    ) -> ChatResult:
        return self._generate(messages = messages, stop = stop)

    def _create_message_dicts(
        self, stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params: Dict[str, Any] = {**{"model": self.model_name}, **self._default_params}
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        return params

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        if "Google Bard encountered an error" in response["content"]:
            raise ValueError("Please review your BARD_SESSION_ID.")
        for res in response["choices"]:
            message = res["content"]
            message=ChatMessage(content = message[0], role = "Bard")
            gen = ChatGeneration(message=message)
            generations.append(gen)
        prompt_tokens = self.prompt_tokens
        completion_tokens = self.get_num_tokens(message.content)
        total_tokens = prompt_tokens + completion_tokens
        token_usage = {"prompt_tokens" : prompt_tokens, "completion_tokens" : completion_tokens, "total_tokens" :total_tokens}
        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

    def get_num_tokens(self, text: str) -> int:
        """Calculate num words"""
        if isinstance(text, list):
            text = text[0]
            if not isinstance(text, str):
                text = text.content
        return len(text)


class BaseBard(BaseLLM):
    """Wrapper around Bard large language models."""

    client: Any  #: :meta private:
    model_name: str = "Bard"
    """Model name to use."""
    max_tokens: int = 256
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""
    frequency_penalty: float = 0
    """Penalizes repeated tokens according to frequency."""
    presence_penalty: float = 0
    """Penalizes repeated tokens."""
    n: int = 1
    """How many completions to generate for each prompt."""
    best_of: int = 1
    """Generates best_of completions server-side and returns the "best"."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    bard_session_id: Optional[str] = None
    batch_size: int = 5000
    """Batch size to use when passing multiple documents to generate."""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to Bard completion API. Default is 600 seconds."""
    max_retries: int = 6

    """
    def __new__(cls, **data: Any) -> Union[BardChat, BaseBard]:  # type: ignore
        '''Initialize the Bard object.'''
        model_name = data.get("model_name", "")
        if model_name.startswith("gpt-3.5-turbo") or model_name.startswith("gpt-4"):
            warnings.warn(
                "You are trying to use a chat model. This way of initializing it is "
                "no longer supported. Instead, please use: "
                "`from langchain.chat_models import ChatOpenAI`"
            )
            return BardChat(**data)
        return super().__new__(cls)
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        disallowed_model_kwargs = all_required_field_names | {"model"}
        invalid_model_kwargs = disallowed_model_kwargs.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        bard_session_id = get_from_dict_or_env(
            values, "bard_session_id", "BARD_SESSION_ID"
        )
        try:
            from Bard import Chatbot

        except ImportError:
            raise ValueError(
                "Could not import GoogleBard python package. "
                "Please install it with `pip install GoogleBard`."
            )
        try:
            values["client"] = Chatbot(bard_session_id)
        except AttributeError:
            raise ValueError(
                "you can't use this BARD_SESSION_ID"
                "Go to https://bard.google.com/"
                "F12 for console"
                "Session: Go to Application → Cookies → __Secure-1PSID. Copy the value of that cookie."
                "Set your env BARD_SESSION_ID = {copied value}"
            )
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Bard API."""
        normal_params = {
            "max_tokens": self.max_tokens,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
            "request_timeout": self.request_timeout,
        }

        # Azure gpt-35-turbo doesn't support best_of
        # don't specify best_of if it is 1
        if self.best_of > 1:
            normal_params["best_of"] = self.best_of

        return {**normal_params, **self.model_kwargs}

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
    ) -> LLMResult:
        """Call out to Bard's endpoint with k unique prompts.

        Args:
            prompts: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The full LLM output.

        Example:
            .. code-block:: python

                response = openai.generate(["Tell me a joke."])
        """
        # TODO: write a unit test for this
        params = self._invocation_params
        sub_prompts = self.get_sub_prompts(params, prompts, stop)
        choices = []
        prompt_tokens = 0
        completion_tokens = 0
        for prompt in sub_prompts:
            prompt = prompt[0]
            if isinstance(prompt, str):
                prompt_tokens += self.get_num_tokens(prompt[0])
            else:                
                prompt_tokens += self.get_num_tokens(prompt[0].content)
        total_tokens = prompt_tokens + completion_tokens
        token_usage: Dict[str, int] = {"prompt_tokens" : prompt_tokens, "completion_tokens" : completion_tokens, "total_tokens" :total_tokens}
        # Includes prompt, completion, and total tokens used.
        for _prompts in sub_prompts:
            response = completion_with_retry(self, prompt=_prompts, **params)
            choices.extend(response["choices"])
        
        return self.create_llm_result(choices, prompts, token_usage)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
    ) -> LLMResult:
        return self._generate(prompts=prompts, stop=stop)

    def get_sub_prompts(
        self,
        params: Dict[str, Any],
        prompts: List[str],
        stop: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """Get the sub prompts for llm call."""
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        if params["max_tokens"] == -1:
            if len(prompts) != 1:
                raise ValueError(
                    "max_tokens set to -1 not supported for multiple inputs."
                )
            params["max_tokens"] = self.max_tokens_for_prompt(prompts[0])
        sub_prompts = [
            prompts[i : i + self.batch_size]
            for i in range(0, len(prompts), self.batch_size)
        ]
        return sub_prompts

    def create_llm_result(
        self, choices: Any, prompts: List[str], token_usage: Dict[str, int]
    ) -> LLMResult:
        """Create the LLMResult from the choices and prompts."""
        generations = []
        for i, _ in enumerate(prompts):
            sub_choices = choices[i * self.n : (i + 1) * self.n]
            generations.append(
                [
                    Generation(
                        text=choice["content"][0],
                        generation_info=dict(
                            id=choice.get("id"),
                        ),
                    )
                    for choice in sub_choices
                ]
            )
        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        return LLMResult(generations=generations, llm_output=llm_output)

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return self._default_params

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "Bard"

    def get_num_tokens(self, text: str) -> int:
        """Calculate num words"""
        if isinstance(text, list):
            text = text[0]
            if not isinstance(text, str):
                text = text.content
        return len(text)

    def modelname_to_contextsize(self, modelname: str) -> int:
        """Calculate the maximum number of tokens possible to generate for a model.

        Args:
            modelname: The modelname we want to know the context size for.

        Returns:
            The maximum context size
        """
        model_token_mapping = {
            "gpt-4": 8192,
            "gpt-4-0314": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-32k-0314": 32768,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-0301": 4096,
            "text-ada-001": 2049,
            "ada": 2049,
            "text-babbage-001": 2040,
            "babbage": 2049,
            "text-curie-001": 2049,
            "curie": 2049,
            "davinci": 2049,
            "text-davinci-003": 4097,
            "text-davinci-002": 4097,
            "code-davinci-002": 8001,
            "code-davinci-001": 8001,
            "code-cushman-002": 2048,
            "code-cushman-001": 2048,
            "Bard": 5000, # NOT 5000 tokens 5000 words can send to Bard
        }

        context_size = model_token_mapping.get(modelname, None)

        if context_size is None:
            raise ValueError(
                f"Unknown model: {modelname}. Please provide a valid OpenAI model name."
                "Known models are: " + ", ".join(model_token_mapping.keys())
            )

        return context_size

    def max_tokens_for_prompt(self, prompt: str) -> int:
        """Calculate the maximum number of tokens possible to generate for a prompt.

        Args:
            prompt: The prompt to pass into the model.

        Returns:
            The maximum number of tokens to generate for a prompt.

        """
        num_tokens = self.get_num_tokens(prompt)

        # get max context size for model by name
        max_size = self.modelname_to_contextsize(self.model_name)
        return max_size - num_tokens


class Bard(BaseBard):
    """Wrapper around Bard large language models.

    To use, you should have the ``GoogleBard`` python package installed, and the
    environment variable ``BARD_SESSION_ID`` set with your API key.

    Any parameters that are valid to be passed to the Bard.Chatbot.ask call can be passed
    in, even if not explicitly saved on this class.
    """

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**{"model": self.model_name}, **super()._invocation_params}

def _create_retry_decorator(llm: Union[BaseBard, BardChat]) -> Callable[[Any], Any]:
    min_seconds = 4
    max_seconds = 15
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    
def completion_with_retry(llm: Union[BaseBard, BardChat], **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm)
    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        if kwargs is not None:
            if "messages" in kwargs.keys():
                messages = get_buffer_string(kwargs["messages"])
            else:
                messages = kwargs["prompt"][0]
        return llm.client.ask(messages)
    return _completion_with_retry(**kwargs)