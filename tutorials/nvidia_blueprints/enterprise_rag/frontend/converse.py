# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the frontend gui for having a conversation."""
import functools
from typing import Any, Dict, List, Tuple

import assets
import fastapi_app_client
import gradio as gr

PATH = "/converse"
TITLE = "Converse"
OUTPUT_TOKENS = 1024
MAX_DOCS = 5

_LOCAL_CSS = """

#contextbox {
    overflow-y: scroll !important;
    max-height: 400px;
}
"""


def build_page(client) -> gr.Blocks:
    """Build the gradio page to be mounted in the frame."""
    kui_theme, kui_styles = assets.load_theme("kaizen")

    with gr.Blocks(title=TITLE, theme=kui_theme, css=kui_styles + _LOCAL_CSS) as page:

        # create the page header
        gr.Markdown(f"# {TITLE}")

        # chat logs
        with gr.Row(equal_height=True):
            chatbot = gr.Chatbot(scale=2, label=client.model_name, type="messages")
            latest_response = gr.Textbox(visible=False)
            context = gr.JSON(
                scale=1,
                label="Knowledge Base Context",
                visible=False,
                elem_id="contextbox",
            )

        # check boxes
        with gr.Row():
            with gr.Column(scale=10, min_width=150):
                kb_checkbox = gr.Checkbox(
                    label="Use knowledge base", info="", value=False
                )

        # text input boxes
        with gr.Row():
            with gr.Column(scale=10, min_width=500):
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press ENTER",
                    container=False,
                )

        # user feedback
        with gr.Row():
            # _ = gr.Button(value="👍  Upvote")
            # _ = gr.Button(value="👎  Downvote")
            # _ = gr.Button(value="⚠️  Flag")
            submit_btn = gr.Button(value="Submit")
            _ = gr.ClearButton(msg)
            _ = gr.ClearButton([msg, chatbot], value="Clear History")
            ctx_show = gr.Button(value="Show Context")
            ctx_hide = gr.Button(value="Hide Context", visible=False)

        # hide/show context
        def _toggle_context(btn: str) -> Dict[gr.component, Dict[Any, Any]]:
            if btn == "Show Context":
                out = [True, False, True]
            if btn == "Hide Context":
                out = [False, True, False]
            return {
                context: gr.update(visible=out[0]),
                ctx_show: gr.update(visible=out[1]),
                ctx_hide: gr.update(visible=out[2]),
            }

        ctx_show.click(_toggle_context, [ctx_show], [context, ctx_show, ctx_hide])
        ctx_hide.click(_toggle_context, [ctx_hide], [context, ctx_show, ctx_hide])

        # form actions
        _my_build_stream = functools.partial(_stream_predict, client)
        msg.submit(
            _my_build_stream,
            [kb_checkbox, msg, chatbot],
            [msg, chatbot, context, latest_response],
        )
        submit_btn.click(
            _my_build_stream,
            [kb_checkbox, msg, chatbot],
            [msg, chatbot, context, latest_response],
        )

    page.queue()
    return page


def _stream_predict(
    client: fastapi_app_client.ChatClient,
    use_knowledge_base: bool,
    question: str,
    chat_history: List[Tuple[str, str]],
) -> Any:
    """Make a prediction of the response to the prompt."""
    if not question.strip():
        return None, chat_history, None, None

    chunks = ""

    # Prepare chat history for the LLM client in the format it expects
    llm_chat_history = []
    if chat_history:
        for message in chat_history:
            # Extract role and content from message, handling both dictionary and tuple formats
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
            elif isinstance(message, tuple) and len(message) == 2:
                role, content = message
            else:
                continue  # Skip invalid formats

            if role and content:
                llm_chat_history.append((role, content))

    # Get documents if using knowledge base
    documents = None
    if use_knowledge_base:
        documents = client.search(prompt=question, num_docs=MAX_DOCS)

    # Create a new messages list for the Gradio component
    messages = list(chat_history) if chat_history else []

    # Add the user's new message
    user_message = {"role": "user", "content": question}
    messages.append(user_message)

    # Add an empty assistant message that we'll update
    assistant_message = {"role": "assistant", "content": ""}
    messages.append(assistant_message)

    # First yield to update UI with user message and empty assistant response
    yield "", messages, documents, ""

    # Stream the response
    try:
        for chunk in client.predict(
            query=question,
            chat_history=llm_chat_history,
            use_knowledge_base=use_knowledge_base,
            num_tokens=OUTPUT_TOKENS,
        ):
            if chunk:
                chunks += chunk
                # Update the assistant's message content
                messages[-1]["content"] = chunks

            yield "", messages, documents, chunks
    except Exception as e:
        # Handle errors gracefully
        error_message = f"Error generating response: {str(e)}"
        messages[-1]["content"] = error_message
        yield "", messages, documents, error_message

    # Clear the input box
    return "", messages, documents, chunks
