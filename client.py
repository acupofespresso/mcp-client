import asyncio
import json
import os
import sys
import time
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv


load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self):
        """Connect to an MCP server
        """
            
        server_params = StdioServerParameters(
            command="uvx",
            args=[
                "mcp-server-fetch",
                "--ignore-robots-txt",
                "--proxy-url", "http://127.0.0.1:7897"
            ],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    def print_streaming_text(self, text: str, delay: float = 0.02):
        """Print text with streaming effect"""
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
    
    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        print("\n🤖 Claude is thinking...")
        
        # Initial Claude API call with streaming
        stream = self.anthropic.messages.create(
            model=os.getenv("ANTHROPIC_MODEL"),
            max_tokens=1000,
            messages=messages,
            tools=available_tools,
            stream=True
        )

        # Process streaming response and handle tool calls
        tool_results = []
        final_text = []
        accumulated_text = ""
        tool_calls = []
        current_tool_input = ""
        
        print("\n💬 ", end="")
        
        for chunk in stream:
            if chunk.type == "message_start":
                continue
            elif chunk.type == "content_block_start":
                if chunk.content_block.type == "text":
                    continue
                elif chunk.content_block.type == "tool_use":
                    tool_calls.append({
                        "id": chunk.content_block.id,
                        "name": chunk.content_block.name,
                        "input": {}
                    })
                    current_tool_input = ""
            elif chunk.type == "content_block_delta":
                if chunk.delta.type == "text_delta":
                    text_chunk = chunk.delta.text
                    accumulated_text += text_chunk
                    self.print_streaming_text(text_chunk, 0.01)
                elif chunk.delta.type == "input_json_delta":
                    if tool_calls:
                        current_tool_input += chunk.delta.partial_json
            elif chunk.type == "content_block_stop":
                # Parse the complete tool input JSON
                if tool_calls and current_tool_input:
                    try:
                        tool_calls[-1]["input"] = json.loads(current_tool_input)
                    except json.JSONDecodeError:
                        tool_calls[-1]["input"] = {}
                current_tool_input = ""
            elif chunk.type == "message_delta":
                continue
            elif chunk.type == "message_stop":
                break
        
        if accumulated_text:
            final_text.append(accumulated_text)
        
        # Handle tool calls if any
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["input"]

            print(f"\n\n🔧 Calling tool {tool_call}...")
            
            # Execute tool call
            result = await self.session.call_tool(tool_name, tool_args)
            tool_results.append({"call": tool_name, "result": result})

            tool_result = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content)
            print(f"\n🔧 Tool {tool_name} returned: \n{tool_result}")

            # Continue conversation with tool results
            messages.append({
                "role": "assistant",
                "content": accumulated_text if accumulated_text else f"I'll use the {tool_name} tool."
            })
            messages.append({
                "role": "user",
                "content": tool_result
            })

            print("\n🤖 Processing tool result...")
            print("\n💬 ", end="")
            
            # Get next response from Claude with streaming
            follow_up_stream = self.anthropic.messages.create(
                model=os.getenv("ANTHROPIC_MODEL"),
                max_tokens=1000,
                messages=messages,
                stream=True
            )
            
            follow_up_text = ""
            for chunk in follow_up_stream:
                if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                    text_chunk = chunk.delta.text
                    follow_up_text += text_chunk
                    self.print_streaming_text(text_chunk, 0.01)
            
            if follow_up_text:
                final_text.append(follow_up_text)
        
        print("\n")
        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    client = MCPClient()
    try:
        await client.connect_to_server()
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())