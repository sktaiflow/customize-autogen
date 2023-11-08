import logging
import time
from typing import List, Optional, Dict, Callable, Union
import sys
from .completion import ChatCompletion
import requests


class AdotChatCompletion(ChatCompletion):

    @classmethod
    def adot_create(
        cls,
        config_list: Optional[List[Dict]] = None,
        **config,
    ):
        """
        입력
        response = oai.ChatCompletion.create(
            context=messages[-1].pop("context", None), messages=self._oai_system_message + messages, **llm_config
        )

        출력

        """
        # Warn if a config list was provided but was empty
        # if type(config_list) is list and len(config_list) == 0:
        #     logger.warning(
        #         "Completion was provided with a config_list, but the list was empty. Adopting default OpenAI behavior, which reads from the 'model' parameter instead."
        #     )

        if config_list:

            ## 첫번쨰 부터 그냥 돌아가는 겁니다.
            for i, each_config in enumerate(config_list):
                prompt_id = each_config.get('prompt_id', False)
                prompt_url = each_config.get('prompt_url', False)
                
                if prompt_id and prompt_url:
                    try:
                        messages = each_config.get('messages', [])

                        history = messages[:-1]
                        message = messages[-1]

                        data = {
                            "promptId": prompt_id,
                            "requestTexts": [
                                {
                                    "key": each_config.get('key', 'content'),
                                    "value": message.get('content', ''),
                                },
                            ],
                            'role': message.get('role', 'user'),
                            'history' : history,
                        }

                        functions = each_config.get('functions', [])
                        if len(functions) > 0:
                            data["functionCall"] = {
                                "control": "auto",
                                "functions" : functions,
                            }

                        headers = {
                            "Accept": "application/json",
                        }
                        resp = requests.post(
                            prompt_url,
                            headers=headers,
                            json=data,
                        )
                        resp = resp.json().get('res', {})

                        ## convert return format
                        choices = []
                        content = resp.get('content', None)
                        function_call = resp.get('functionCall', None)
                        
                        if content is not None:
                            choices.append({
                                "message" : {
                                    'role':'assistant',
                                    'content':content
                                },
                            })

                        elif function_call is not None:
                            choices.append({
                                "message" : {
                                    'role': 'assistant',
                                    'function_call': function_call,
                                },
                            })

                        return_dict = {
                            "id": resp.get("transactionId", ""),
                            "model": resp.get('model', ""),
                            "choices": choices,
                            "usage" : resp.get("usage", {}),
                        }
                        return return_dict
                    except Exception as e:
                        print(f"[SKT AZURE ERROR] {e}")

        ## 없으면 class create로 return            
        return cls.create(config_list=config_list, **config)



     

