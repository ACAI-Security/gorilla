import requests
import time
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.model_style import ModelStyle
from bfcl_eval.model_handler.utils import (
    func_doc_language_specific_pre_processing,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
)
from overrides import override


class SOHandler(BaseHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI_Responses
        self.base_url = "http://localhost:8000"

    @override
    def _query_prompting(self, inference_data: dict):
        messages = inference_data["message"]
        functions = inference_data["function"]
        
        # New format: wrap function details in "function" object
        tools = [{
            "type": "function",
            "function": {
                "name": f["name"], 
                "description": f["description"], 
                "parameters": f["parameters"]
            },
            "strict": True
        } for f in functions]
        
        request_data = {"model": "", "type": "start", "messages": messages}
        if tools:
            request_data["tools"] = tools
            
        start_time = time.time()
        response = requests.post(f"{self.base_url}/v1/chat/completions", json=request_data, timeout=30)
        end_time = time.time()
        
        return response, end_time - start_time

    @override
    def _parse_query_response_prompting(self, api_response) -> dict:
        response_json = api_response.json()
        
        if "choices" in response_json:
            message = response_json["choices"][0]["message"]
            
            # Check if there's a function_call in the response
            if "function_call" in message and message["function_call"]:
                # Return the function call in the expected format
                func_call = message["function_call"]
                msg_content = message.get('content', '').replace("'", "\\'")  # Escape single quotes
                content = f"ResponseFnCall(value={{'function_call': {{'name': '{func_call['name']}', 'arguments': '{func_call['arguments']}'}}, 'message': '{msg_content}'}}, info='{msg_content}')"
            else:
                # No function call, just return content
                content = message.get("content", "")
        else:
            content = str(response_json)
        
        return {
            "model_responses": content,
            "input_token": response_json.get("usage", {}).get("prompt_tokens", 0),
            "output_token": response_json.get("usage", {}).get("completion_tokens", 0),
        }

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions = func_doc_language_specific_pre_processing(test_entry["function"], test_entry["id"].rsplit("_", 1)[0])
        return {"message": [], "function": functions}

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
    def _add_execution_results_prompting(self, inference_data: dict, execution_results: list[str], model_response_data: dict) -> dict:
        for result, decoded_response in zip(execution_results, model_response_data["model_responses_decoded"]):
            inference_data["message"].append({"role": "tool", "name": decoded_response, "content": result})
        return inference_data

    @override
    def decode_ast(self, result, language="Python"):
        import json
        import ast
        
        # Handle different error cases
        if isinstance(result, dict) and 'info' in result:
            if 'JSON serializable' in result['info'] or 'Execution failed' in result['info']:
                return []
        
        # Handle string dict format for serialization errors: "{'info': 'Stopped because Object of type SOInt is not JSON serializable'}"
        if isinstance(result, str) and result.startswith("{'info':") and 'JSON serializable' in result:
            return []
        
        # Handle ResponseError cases
        if isinstance(result, str) and result.startswith("ResponseError("):
            return []
        
        # Handle ResponseFnCall format: ResponseFnCall(value={'function_call': {'name': '...', 'arguments': '...'}}, info="...")
        if isinstance(result, str) and result.startswith("ResponseFnCall("):
            try:
                # Use regex to extract function name and arguments more reliably
                import re
                
                # Extract function name
                name_match = re.search(r"'name':\s*'([^']+)'", result)
                if not name_match:
                    print("Debug: Could not extract function name")
                    return []
                
                name = name_match.group(1)
                
                # Extract arguments JSON string
                args_match = re.search(r"'arguments':\s*'([^']+)'", result)
                if not args_match:
                    args_match = re.search(r"'arguments':\s*\"([^\"]+)\"", result)
                
                if not args_match:
                    print("Debug: Could not extract arguments")
                    return []
                
                args_str = args_match.group(1)
                # Unescape any escaped quotes
                args_str = args_str.replace('\\"', '"').replace("\\'", "'")
                
                # Parse arguments JSON string
                args_dict = json.loads(args_str)
                
                # Convert integer 0/1 to boolean false/true for BFCL compatibility
                for key, value in args_dict.items():
                    if isinstance(value, int) and value in [0, 1]:
                        # Check if this might be a boolean by looking at parameter name patterns
                        boolean_patterns = ['formatted', 'detailed', 'include', 'specific', 'enabled', 'active', 'visible', 'required', 'show', 'display', 'verbose', 'debug']
                        if any(pattern in key.lower() for pattern in boolean_patterns):
                            args_dict[key] = bool(value)
                
                # Return in BFCL expected format: [{"function_name": {"param": "value"}}]
                return [{name: args_dict}]
                
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"Debug: Failed to parse ResponseFnCall: {e}")
                return []
        
        # Fallback to default decoder
        return default_decode_ast_prompting(result, language)

    @override
    def decode_execute(self, result):
        return default_decode_execute_prompting(result)

    # FC methods - minimal implementations
    @override
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    @override
    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions = func_doc_language_specific_pre_processing(test_entry["function"], test_entry["id"].rsplit("_", 1)[0])
        # Store raw functions for _query_FC to format properly
        inference_data["tools"] = functions
        return inference_data

    @override
    def _query_FC(self, inference_data: dict):
        messages = inference_data["message"]
        tools = inference_data.get("tools", [])
        
        # Tools only sent on first request (type: "start")
        is_first_request = len(messages) <= 1
        
        # New format: wrap function details in "function" object
        formatted_tools = [{
            "type": "function",
            "function": {
                "name": tool["name"], 
                "description": tool["description"], 
                "parameters": tool["parameters"]
            },
            "strict": True
        } for tool in tools] if tools else []
        
        request_data = {
            "model": "gpt-4", 
            "type": "start" if is_first_request else "continue",
            "messages": messages
        }
        
        if is_first_request and formatted_tools:
            request_data["tools"] = formatted_tools
            
        start_time = time.time()
        response = requests.post(f"{self.base_url}/v1/chat/completions", json=request_data, timeout=30)
        end_time = time.time()
        
        return response, end_time - start_time

    @override
    def _parse_query_response_FC(self, api_response) -> dict:
        return self._parse_query_response_prompting(api_response)

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
        return self._add_execution_results_prompting(inference_data, execution_results, model_response_data)