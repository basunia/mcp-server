import asyncio
import os
import re
import sys
import json
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from groq import Groq, APIStatusError
from dotenv import load_dotenv

load_dotenv()

# openai/gpt-oss-120b is noticeably more reliable at well-formed tool calls
# on Groq than the Llama tool-use models (fewer malformed-argument errors).
GROQ_MODEL = "openai/gpt-oss-120b"


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # Persisted across turns so follow-up questions ("for which location
        # you gave...") have context. Reset with the 'reset' chat command.
        self.messages = [
            {
                "role": "system",
                "content": (
                    "When calling tools, always pass numeric parameters as JSON "
                    "numbers (e.g. 37.7749), never as quoted strings."
                ),
            }
        ]

    async def connect_to_server(self, server_script_path: str):
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = sys.executable if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        print(f"server_params: {server_params}", file=sys.stderr)

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    def _mcp_tools_to_groq_format(self, mcp_tools):
        """Convert MCP tool schema (name/description/inputSchema) into the
        OpenAI-style function-calling format Groq expects."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in mcp_tools
        ]

    def _coerce_args(self, args: dict, schema: dict) -> dict:
        """Cast string values back to the numeric/boolean types the tool's
        JSON schema expects. Llama tool-use models on Groq sometimes quote
        numbers (e.g. "37.7749" instead of 37.7749)."""
        props = schema.get("properties", {})
        fixed = {}
        for key, value in args.items():
            expected = props.get(key, {}).get("type")
            if isinstance(value, str):
                if expected in ("number", "integer"):
                    try:
                        value = int(value) if expected == "integer" else float(value)
                    except ValueError:
                        pass
                elif expected == "boolean":
                    value = value.strip().lower() in ("true", "1", "yes")
            fixed[key] = value
        return fixed

    def _recover_from_tool_use_failed(self, error: APIStatusError, available_tools: list):
        """Groq validates tool arguments server-side and rejects the whole
        request if types don't match the schema, instead of letting the
        model retry. When that happens, pull the raw generation out of the
        error body, fix the types ourselves, and run the tool anyway.
        Returns (tool_name, args) on success, or None if unrecoverable."""
        body = getattr(error, "body", None) or {}
        err = body.get("error", {}) if isinstance(body, dict) else {}
        if err.get("code") != "tool_use_failed":
            return None

        raw = err.get("failed_generation", "")
        # Llama models sometimes drop the '>' between the name and the JSON
        # args (e.g. <function=get_alerts{"state": "FL"}</function>), so
        # treat it as optional rather than requiring the well-formed tag.
        match = re.search(r"<function=(\w+)>?\s*(\{.*\})\s*</function>", raw)
        if not match:
            return None

        tool_name, raw_args = match.group(1), match.group(2)
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            return None

        schema = next(
            (t["function"]["parameters"] for t in available_tools if t["function"]["name"] == tool_name),
            {},
        )
        return tool_name, self._coerce_args(args, schema)

    async def process_query(self, query: str) -> str:
        self.messages.append({"role": "user", "content": query})

        response = await self.session.list_tools()
        available_tools = self._mcp_tools_to_groq_format(response.tools)

        final_text = []
        MAX_TOOL_ITERATIONS = 6  # hard cap so a confused model can't loop forever
        iterations = 0

        # Loop so the model can chain multiple tool calls if needed
        while True:
            iterations += 1
            if iterations > MAX_TOOL_ITERATIONS:
                final_text.append(
                    "[Stopped after too many tool calls without a final answer — "
                    "try asking a more specific question.]"
                )
                break

            try:
                completion = self.groq.chat.completions.create(
                    model=GROQ_MODEL,
                    max_tokens=1000,
                    messages=self.messages,
                    tools=available_tools,
                )
            except APIStatusError as e:
                recovered = self._recover_from_tool_use_failed(e, available_tools)
                if recovered is None:
                    raise  # some other error - let it surface normally

                tool_name, tool_args = recovered
                final_text.append(
                    f"[Calling tool {tool_name} with args {tool_args}] (auto-corrected argument types)"
                )

                result = await self.session.call_tool(tool_name, tool_args)
                result_text = "\n".join(
                    block.text for block in result.content if hasattr(block, "text")
                )

                # Groq rejected the call before the model's turn was recorded,
                # so splice the tool result in as a user message and ask again.
                self.messages.append(
                    {"role": "user", "content": f"Tool '{tool_name}' returned: {result_text}"}
                )
                continue

            msg = completion.choices[0].message

            if msg.content:
                final_text.append(msg.content)

            if not msg.tool_calls:
                break  # model gave a final answer, no more tools needed

            # Record the assistant's tool-call request in the conversation
            self.messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            # Execute each requested tool via MCP, feed the result back
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                result = await self.session.call_tool(tool_name, tool_args)
                result_text = "\n".join(
                    block.text for block in result.content if hasattr(block, "text")
                )

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    }
                )

        return "\n".join(final_text)

    async def chat_loop(self):
        print("\nMCP Client Started!")
        print("Type your queries, 'reset' to clear history, or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                if query.lower() == 'reset':
                    self.messages = self.messages[:1]  # keep only the system message
                    print("\nConversation history cleared.")
                    continue

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
