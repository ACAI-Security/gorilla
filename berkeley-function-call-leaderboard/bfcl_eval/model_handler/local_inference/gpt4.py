import requests
import time
import json
import os
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.model_style import ModelStyle
from overrides import override


class GPT4Handler(BaseHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI_Responses
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"

    def _make_request(self, messages, tools=None):
        # Use the OpenRouter format for GPT-4o
        request_data = {
            "model": "openai/gpt-4o-2024-11-20",
            "messages": messages,
            "temperature": self.temperature
        }
        
        if tools:
            request_data["tools"] = tools
        
        
        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=request_data,
            timeout=30
        )
        end_time = time.time()
        
        
        return response, end_time - start_time

    def _parse_response(self, api_response):
        response_json = api_response.json()
        
        content = ""
        if "choices" in response_json and response_json["choices"]:
            message = response_json["choices"][0]["message"]
            content = message.get("content", "")
        
        return {
            "model_responses": content,
            "input_token": response_json.get("usage", {}).get("prompt_tokens", 0),
            "output_token": response_json.get("usage", {}).get("completion_tokens", 0),
        }

    @override
    def _query_prompting(self, inference_data: dict):
        return self._make_request(inference_data["message"])

    @override
    def _parse_query_response_prompting(self, api_response) -> dict:
        return self._parse_response(api_response)

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        return {"message": []}

    @override
    def add_first_turn_message_prompting(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _add_next_turn_user_message_prompting(self, inference_data: dict, user_message: list[dict]) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    @override
    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].append({"role": "assistant", "content": model_response_data["model_responses"]})
        return inference_data

    @override
    def decode_ast(self, result, language="Python"):
        import json
        if "FC" in self.model_name or self.is_fc_model:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                decoded_output.append({name: params})
            return decoded_output
        else:
            from bfcl_eval.model_handler.utils import default_decode_ast_prompting
            return default_decode_ast_prompting(result, language)

    @override
    def decode_execute(self, result):
        if "FC" in self.model_name or self.is_fc_model:
            from bfcl_eval.model_handler.utils import convert_to_function_call
            return convert_to_function_call(result)
        else:
            from bfcl_eval.model_handler.utils import default_decode_execute_prompting
            return default_decode_execute_prompting(result)

    @override
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        return {"message": []}

    @override
    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        from bfcl_eval.model_handler.utils import func_doc_language_specific_pre_processing, _cast_to_openai_type
        from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
        
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]
        
        functions = func_doc_language_specific_pre_processing(functions, test_category)
        
        # Build tools with sanitized names and store mapping for this request
        tools = []
        self.current_function_mapping = {}
        
        for func in functions:
            original_name = func["name"]
            sanitized_name = func["name"].replace(".", "_")
            
            # Store mapping for response parsing
            self.current_function_mapping[sanitized_name] = original_name
            
            # Use proper OpenAI nested structure
            tool = {
                "type": "function",
                "function": {
                    "name": sanitized_name,
                    "description": func["description"],
                    "parameters": {
                        "type": "object",
                        "properties": _cast_to_openai_type(func["parameters"]["properties"].copy(), GORILLA_TO_OPENAPI),
                        "required": func["parameters"].get("required", [])
                    }
                }
            }
            tools.append(tool)
        
        
        inference_data["tools"] = tools
        return inference_data

    @override
    def _query_FC(self, inference_data: dict):
        messages = inference_data["message"]
        tools = inference_data.get("tools", [])
        
        # Convert BFCL tool format to OpenAI API format
        openai_tools = []
        for tool in tools:
            if tool.get("type") == "function" and "name" in tool:
                # Convert from BFCL format to OpenAI format
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                }
                openai_tools.append(openai_tool)
            else:
                # Already in correct format
                openai_tools.append(tool)
        
        return self._make_request(messages, openai_tools)

    @override
    def _parse_query_response_FC(self, api_response) -> dict:
        response_json = api_response.json()
        
        try:
            if "choices" in response_json and response_json["choices"]:
                message = response_json["choices"][0]["message"]
                
                if "tool_calls" in message and message["tool_calls"]:
                    # Parse tool calls and map sanitized names back to originals
                    model_responses = []
                    original_function_names = []
                    
                    for func_call in message["tool_calls"]:
                        sanitized_name = func_call["function"]["name"]
                        original_name = getattr(self, 'current_function_mapping', {}).get(sanitized_name, sanitized_name)
                        
                        model_responses.append({original_name: func_call["function"]["arguments"]})
                        # Store original name for the JSON result
                        original_function_names.append(original_name if original_name != sanitized_name else "")
                else:
                    model_responses = message.get("content", "")
                    original_function_names = []
            else:
                model_responses = str(response_json)
                original_function_names = []
        except:
            model_responses = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            original_function_names = []
        
        result = {
            "model_responses": model_responses,
            "input_token": response_json.get("usage", {}).get("prompt_tokens", 0),
            "output_token": response_json.get("usage", {}).get("completion_tokens", 0),
        }
        
        # Add original function names if we have any renames
        if original_function_names:
            result["original_function_names"] = original_function_names
        
        return result

    @override
    def add_first_turn_message_FC(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        return self.add_first_turn_message_prompting(inference_data, first_turn_message)

    @override
    def _add_next_turn_user_message_FC(self, inference_data: dict, user_message: list[dict]) -> dict:
        return self._add_next_turn_user_message_prompting(inference_data, user_message)

    @override
    def _add_assistant_message_FC(self, inference_data: dict, model_response_data: dict) -> dict:
        return self._add_assistant_message_prompting(inference_data, model_response_data)

    @override
    def _add_execution_results_FC(self, inference_data: dict, execution_results: list[str], model_response_data: dict) -> dict:
        return inference_data