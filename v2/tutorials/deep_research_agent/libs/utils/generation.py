import re

import flyte
from libs.utils.llms import single_shot_llm_call


@flyte.trace
async def generate_toc_image(prompt: str, planning_model: str, topic: str) -> str:
    """Generate a table of contents image"""
    from together import Together

    image_generation_prompt = single_shot_llm_call(
        model=planning_model, system_prompt=prompt, message=f"Research Topic: {topic}"
    )

    if image_generation_prompt is None:
        raise ValueError("Image generation prompt is None")

    # HERE WE CALL THE TOGETHER API SINCE IT'S AN IMAGE GENERATION REQUEST
    client = Together()
    imageCompletion = client.images.generate(
        model="black-forest-labs/FLUX.1-dev",
        width=1024,
        height=768,
        steps=28,
        prompt=image_generation_prompt,
    )

    return imageCompletion.data[0].url  # type: ignore


@flyte.trace
async def generate_html(markdown_content: str, toc_image_url: str) -> str:
    """
    Generate an HTML report from markdown formatted content.
    Returns the generated HTML as a string.
    """
    try:
        import datetime

        import markdown
        from pymdownx.superfences import SuperFencesCodeExtension

        year = datetime.datetime.now().year
        month = datetime.datetime.now().strftime("%B")
        day = datetime.datetime.now().day

        # Extract title from first line if not provided
        lines = markdown_content.split("\n")
        title = lines[0].strip("# ")

        content = markdown_content

        # Convert markdown to HTML with table support
        html_body = markdown.markdown(
            content,
            extensions=[
                "tables",
                "fenced_code",
                SuperFencesCodeExtension(
                    custom_fences=[
                        {"name": "mermaid", "class": "mermaid", "format": fence_mermaid}
                    ]
                ),
            ],
        )

        # Add mermaid header
        mermaid_header = """<script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({startOnLoad: true });
            </script>"""

        # Directly parse HTML to extract headings and build TOC
        heading_pattern = re.compile(r'<h([2-3])(?:\s+id="([^"]+)")?>([^<]+)</h\1>')
        toc_items = []
        section_count = 0
        subsection_counts = {}

        # First pass: Add IDs to all headings that don't have them
        modified_html = html_body
        for match in heading_pattern.finditer(html_body):
            level = match.group(1)
            heading_id = match.group(2)
            heading_text = match.group(3)

            # If heading doesn't have an ID, create one and update the HTML
            if not heading_id:
                heading_id = re.sub(r"[^\w\-]", "-", heading_text.lower())
                heading_id = re.sub(r"-+", "-", heading_id).strip("-")

                # Replace the heading without ID with one that has an ID
                original = f"<h{level}>{heading_text}</h{level}>"
                replacement = f'<h{level} id="{heading_id}">{heading_text}</h{level}>'
                modified_html = modified_html.replace(original, replacement)

        # Update the HTML body with the added IDs
        html_body = modified_html

        # Second pass: Build the TOC items
        for match in heading_pattern.finditer(modified_html):
            level = match.group(1)
            heading_id = match.group(2) or re.sub(
                r"[^\w\-]", "-", match.group(3).lower()
            )
            heading_text = match.group(3)

            if level == "2":  # Main section (h2)
                section_count += 1
                subsection_counts[section_count] = 0
                toc_items.append(
                    f'<a class="nav-link py-1" href="#{heading_id}">{section_count}. {heading_text}</a>'
                )
            elif level == "3":  # Subsection (h3)
                parent_section = section_count
                subsection_counts[parent_section] += 1
                subsection_num = subsection_counts[parent_section]
                toc_items.append(
                    f'<a class="nav-link py-1 ps-3" href="#{heading_id}">{parent_section}.{subsection_num}. {heading_text}</a>'
                )

        current_date = datetime.datetime.now().strftime("%B %Y")

        # Create a complete HTML document with enhanced styling and structure
        html_doc = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <base target="_self">
            <meta name="author" content="Research Report">
            <meta name="description" content="Comprehensive research report">
            <!-- Bootstrap CSS -->
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <!-- DataTables CSS -->
            <link href="https://cdn.datatables.net/1.13.5/css/dataTables.bootstrap5.min.css" rel="stylesheet">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
            {mermaid_header}
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    color: #2c3e50;
                    margin-top: 1.5em;
                    margin-bottom: 0.5em;
                }}
                h1 {{
                    font-size: 2.2em;
                    border-bottom: 2px solid #e0e0e0;
                    padding-bottom: 0.5em;
                    margin-bottom: 0.7em;
                }}
                h2 {{
                    font-size: 1.8em;
                    border-bottom: 1px solid #e0e0e0;
                    padding-bottom: 0.3em;
                    margin-top: 2em;
                }}
                p {{
                    margin: 1em 0;
                }}
                code {{
                    background-color: #f6f8fa;
                    border-radius: 3px;
                    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
                    font-size: 85%;
                    padding: 0.2em 0.4em;
                }}
                pre {{
                    background-color: #f6f8fa;
                    border-radius: 3px;
                    padding: 16px;
                    overflow: auto;
                }}
                pre code {{
                    background-color: transparent;
                    padding: 0;
                }}
                blockquote {{
                    background-color: #f8f9fa;
                    border-left: 4px solid #6c757d;
                    padding: 1em;
                    margin: 1.5em 0;
                }}
                ul, ol {{
                    padding-left: 2em;
                }}
                /* Table styling for non-DataTables tables */
                table:not(.dataTable) {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 1em 0;
                    overflow-x: auto;
                    display: block;
                }}
                table:not(.dataTable) th, table:not(.dataTable) td {{
                    border: 1px solid #dfe2e5;
                    padding: 8px 13px;
                    text-align: left;
                }}
                table:not(.dataTable) th {{
                    background-color: #f0f0f0;
                    font-weight: 600;
                    border-bottom: 2px solid #ddd;
                }}
                table:not(.dataTable) tr:nth-child(even) {{
                    background-color: #f8f8f8;
                }}
                a {{
                    color: #0366d6;
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
                img {{
                    max-width: 100%;
                }}
                .container {{
                    background-color: white;
                    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                    padding: 40px;
                }}
                /* Cursor for sortable columns */
                .dataTable thead th {{
                    cursor: pointer;
                }}
                .toc-container {{
                    background-color: #f9f9f9;
                    border-left: 3px solid #0366d6;
                    padding: 15px;
                    border-radius: 4px;
                }}
                .toc-container h5 {{
                    margin-top: 0;
                    margin-bottom: 0.7em;
                }}
                .nav-link {{
                    color: #0366d6;
                    font-size: 0.95em;
                    padding-top: 0.2em;
                    padding-bottom: 0.2em;
                }}
                .report-header {{
                    margin-bottom: 2em;
                }}
                /* Enhanced table styling */
                table:not(.dataTable) th {{
                    background-color: #f0f0f0;
                    font-weight: 600;
                    border-bottom: 2px solid #ddd;
                }}
                /* Print and citation buttons */
                .report-actions {{
                    margin: 2em 0;
                    text-align: right;
                }}
                .print-button, .cite-button {{
                    background-color: #f8f9fa;
                    border: 1px solid #ddd;
                    padding: 8px 15px;
                    border-radius: 4px;
                    margin-left: 10px;
                    color: #495057;
                    text-decoration: none;
                    display: inline-flex;
                    align-items: center;
                    font-size: 0.9em;
                }}
                .print-button:hover, .cite-button:hover {{
                    background-color: #e9ecef;
                    text-decoration: none;
                }}
                .print-button i, .cite-button i {{
                    margin-right: 5px;
                }}
                /* Academic citation formatting */
                ol.references li {{
                    margin-bottom: 1em;
                    padding-left: 1.5em;
                    text-indent: -1.5em;
                }}
            .report-header img {{
                    max-width: 100%;
                    height: auto;
                    margin-bottom: 1.5em;
                    border-radius: 4px;
                }}
            .audio-container {{
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 20px 0;
                    border: 1px solid #e0e0e0;
                    display: flex;
                    align-items: center;
                }}
            .audio-container i {{
                    font-size: 1.8em;
                    color: #0366d6;
                    margin-right: 15px;
                }}
            .audio-container audio {{
                    width: 100%;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="report-header mb-4">
                    <div class="text-end text-muted mb-2">
                        <small>Research Report | Published: {current_date}</small>
                    </div>
                    {f'<img src="{toc_image_url}" alt="Report Header Image">' if toc_image_url else ''}
                    <h1>{title}</h1>
                    <div class="author-info mb-3">
                        <p class="text-muted">
                            <strong>Disclaimer:</strong>
                            This AI-generated report may contain hallucinations, bias, or inaccuracies.
                            Always verify information from independent sources before making
                            decisions based on this content.
                        </p>
                    </div>
                </div>
                <div class="report-actions">
                    <a href="#" class="cite-button" onclick="showCitation(); return false;">
                        <i class="bi bi-quote"></i> Cite
                    </a>
                    <a href="#" class="print-button" onclick="window.print(); return false;">
                        <i class="bi bi-printer"></i> Print
                    </a>
                </div>
                <div class="toc-container mb-4">
                    <h5>Table of Contents</h5>
                    <nav id="toc" class="nav flex-column">
                        {"".join(toc_items)}
                    </nav>
                </div>
                {html_body}
            </div>

            <!-- jQuery -->
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <!-- Bootstrap JS Bundle with Popper -->
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
            <!-- DataTables JS -->
            <script src="https://cdn.datatables.net/1.13.5/js/jquery.dataTables.min.js"></script>
            <script src="https://cdn.datatables.net/1.13.5/js/dataTables.bootstrap5.min.js"></script>
            <script>
                function showCitation() {{
                    const today = new Date();
                    const year = today.getFullYear();
                    const month = today.toLocaleString('default', {{ month: 'long' }});
                    const day = today.getDate();
                    alert(`Research Report. ({year}). {title}. Retrieved {month} {day}, {year}.`);
                }}

                // Initialize all tables with DataTables
                $(document).ready(function() {{
                    $('table').each(function() {{
                        $(this).addClass('table table-striped table-bordered');
                        $(this).DataTable({{
                            responsive: true,
                            paging: $(this).find('tr').length > 10,
                            searching: $(this).find('tr').length > 10,
                            info: $(this).find('tr').length > 10,
                            order: [] // Disable initial sort
                        }});
                    }});
                }});
            </script>
        </body>
        </html>
        """

        return html_doc

    except Exception as error:
        error_msg = f"HTML conversion failed: {str(error)}"
        raise Exception(error_msg)


def fence_mermaid(source, language, css_class, options, md, **kwargs):
    """Clean and process mermaid code blocks."""
    # Filter out title lines and clean whitespace
    cleaned_lines = [
        line.rstrip() for line in source.split("\n") if "title" not in line
    ]
    cleaned_source = "\n".join(cleaned_lines).strip()

    return f'<div class="mermaid">{cleaned_source}</div>'
