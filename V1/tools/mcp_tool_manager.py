import requests
import os

# Example of supported MCP providers
MCP_PROVIDERS = {
    "zapier": {
        "api_url": "https://zapier.com/api/mcp/tools",
        "api_key_env": "ZAPIER_API_KEY",
    },
    "power_automate": {
        "api_url": "https://api.powerautomate.com/mcp/tools",
        "api_key_env": "POWER_AUTOMATE_API_KEY",
    },
    "make": {
        "api_url": "https://www.integromat.com/api/mcp/tools",
        "api_key_env": "MAKE_API_KEY",
    },
    "aws_step_functions": {
        "api_url": "https://api.aws.com/mcp/tools",
        "api_key_env": "AWS_MCP_API_KEY",
    },
}

def get_mcp_tool(tool_name: str):
    """
    Check all MCP providers for the requested tool.
    
    :param tool_name: The name of the MCP tool to look up.
    :return: A dictionary with tool details if found, otherwise None.
    """
    for provider, details in MCP_PROVIDERS.items():
        api_url = details["api_url"]
        api_key = os.getenv(details["api_key_env"])  # Get API key from env variable

        if not api_key:
            continue  # Skip if no API key is set

        try:
            response = requests.get(
                f"{api_url}/{tool_name}",
                headers={"Authorization": f"Bearer {api_key}"}
            )

            if response.status_code == 200:
                tool_data = response.json()
                return {
                    "provider": provider,
                    "tool_name": tool_data.get("name"),
                    "execution_url": tool_data.get("execution_url"),
                    "api_key": api_key,  # Pass API key for execution if needed
                }

        except requests.RequestException as e:
            print(f"Error accessing {provider} MCP API: {e}")

    return None  # No tool found