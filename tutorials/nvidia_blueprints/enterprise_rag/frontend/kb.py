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
"""This module contains the frontend gui for chat."""
import asyncio

import assets
import fastapi_app_client
import gradio as gr
import union

PATH = "/kb"
TITLE = "Knowledge Base Management"


def build_page(client: fastapi_app_client.ChatClient) -> gr.Blocks:
    """Buiild the gradio page to be mounted in the frame."""
    kui_theme, kui_styles = assets.load_theme("kaizen")

    with gr.Blocks(title=TITLE, theme=kui_theme, css=kui_styles) as page:
        # create the page header
        gr.Markdown(f"# {TITLE}")

        with gr.Row():
            upload_button = gr.UploadButton("Add File", file_count="multiple")
        with gr.Row():
            file_paths = gr.Files()

        progress_bars = gr.Row()
        progress = {}

        async def track_progress(execution_id):
            """Track execution progress and update the progress bar."""
            remote = union.UnionRemote()

            execution = remote.fetch_execution(name=execution_id)

            if execution_id not in progress:
                progress[execution_id] = gr.Progress()
                progress_bars.add(progress[execution_id])

            progress[execution_id](0, desc=f"🚀 Running workflow for {execution_id}...")

            progress_value = 0.1
            while True:
                execution = remote.sync_execution(execution)

                if execution.is_done:
                    if execution.error:
                        url = remote.generate_console_url(execution)
                        progress[execution_id](
                            1,
                            desc=f"❌ Error: {execution.error.message}\nExecution URL: {url}",
                        )
                        await asyncio.sleep(10)
                        return

                    progress[execution_id](1, desc=f"✅ File uploaded to vector DB!")
                    await asyncio.sleep(10)
                    return

                progress_value = min(progress_value + 0.1, 0.9)
                progress[execution_id](
                    progress_value, desc=f"⏳ Processing {execution_id}..."
                )
                await asyncio.sleep(2)

        async def start_executions(files, version):
            """Start executions and track progress for multiple files."""
            file_paths, execution_ids = client.upload_documents(files, version=version)

            tasks = [track_progress(eid) for eid in execution_ids]
            await asyncio.gather(*tasks)
            return file_paths

        with gr.Row():
            files_df = gr.Dataframe(
                headers=["File Uploaded"],
                datatype=["str"],
                col_count=(1, "fixed"),
                value=lambda: client.get_uploaded_documents(),
                every=5,
            )

        with gr.Row():
            buffer_textbox = gr.Textbox(
                label="Name of the document to delete", interactive=True, visible=True
            )
            message_textbox = gr.Textbox(
                label="Message", interactive=False, visible=True
            )

        with gr.Row():
            delete_button = gr.Button("Delete")

        with gr.Row():
            version_textbox = gr.Textbox(
                label="Workflow Version",
                value="latest",
                interactive=True,
            )

        upload_button.upload(
            start_executions,
            [upload_button, version_textbox],
            [file_paths],
        )

        # Files dataframe action
        files_df.select(
            return_selected_file, inputs=[files_df], outputs=[buffer_textbox]
        )

        # Delete button action
        delete_button.click(
            fn=client.delete_documents, inputs=buffer_textbox, outputs=message_textbox
        )

    page.queue()
    return page


def return_selected_file(selected_index: gr.SelectData, dataframe):
    """Returns selected files from DataFrame"""
    if selected_index:
        val = dataframe.iloc[selected_index.index[0]]
        dataframe = dataframe.drop(selected_index.index[0])
        return val.iloc[0]
    return None
