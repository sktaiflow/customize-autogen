import re
from autogen.agentchat.agent import Agent
from autogen.agentchat import AdotUserProxyAgent
from jsonschema import validate, ValidationError, SchemaError

from typing import Callable, Dict, Optional, Union, List, Tuple, Any
from IPython import get_ipython
import json

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x



class TemplateUserProxyAgent(AdotUserProxyAgent):
    def __init__(
        self,
        name: str,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "ALWAYS",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Optional[Union[Dict, bool]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
        llm_config: Optional[Union[Dict, bool]] = False,
        system_message: Optional[str] = "",
        output_schema:Dict=None,
        output_template:Dict=None,
        database=None,
        convert_schema_func=None,
    ):
        super().__init__(
            name,
            is_termination_msg,
            max_consecutive_auto_reply,
            human_input_mode,
            function_map,
            code_execution_config,
            default_auto_reply,
            llm_config,
            system_message,
            database,
        )
        
        ## override
        self._reply_func_list = []
        self.register_reply([Agent, None], TemplateUserProxyAgent.generate_oai_reply)
        self.register_reply([Agent, None], TemplateUserProxyAgent.generate_code_execution_reply)
        self.register_reply([Agent, None], TemplateUserProxyAgent.generate_function_call_reply)
        self.register_reply([Agent, None], TemplateUserProxyAgent.check_termination_and_human_reply)
        self.register_reply([Agent, None], TemplateUserProxyAgent.check_json_template)
        
        if output_schema is None:
            raise ValueError("None value in output_schema")
        self._output_schema = output_schema
        if output_template is None:
            output_template = {key:value['description'] for key, value in output_schema['properties'].items()}
        self._output_template = output_template

        self._convert_schema_func=convert_schema_func

    def check_json_template(
        self,
        session_id:str,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        if messages is None:
            messages = self._oai_messages.query(session_id, agent=sender)
        message = messages[-1]
        content = message.get("content", "")

        output_schema = self._output_schema
        if self._convert_schema_func is not None:
            output_schema = self._convert_schema_func(session_id=session_id, output_schema=output_schema)
        
        if content is not None and content.rstrip() != "":
            try:
                json_message = eval(content)
                validate(instance=json_message, schema=output_schema,)
            except SchemaError as e:
                raise SchemaError(f"Output Schema is invalid [{e.message}]")
            except ValidationError as e:
                return_content = f"{e.message}\nThe output should be formatted as JSON instance that conforms to the JSON schema below:\n{str(self._output_template)}"
                return True, return_content
            except Exception as e:
                return_content = f"The output format is incorrect.\nThe output should be formatted as JSON instance that conforms to the JSON schema below:\n{str(self._output_template)}\n\nRespond or Request using json format."
                return True, return_content
        return False, None


