import os
import json
import re
import tiktoken
from langchain_text_splitters import RecursiveJsonSplitter

parent_directory = os.path.join(os.path.dirname(__file__), "Diagnosis_flowchart")

def load_json_files(folder_path):
    json_data = []
    # Create dir if not exists for safety
    os.makedirs(folder_path, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                json_data.append(data)
    return json_data

data_list = load_json_files(parent_directory)

json_splitter = RecursiveJsonSplitter(max_chunk_size=240)
docs = json_splitter.create_documents(texts=data_list) if data_list else []

def chunk_to_markdown(json_text, level=1):
    """
    Convert a JSON chunk (string) into hierarchical Markdown.
    """
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return json_text
        
    def recursive_to_md(data, lvl):
        md = ""
        if isinstance(data, dict):
            for key, value in data.items():
                md += f"{'#' * lvl} {key}\n\n"
                md += recursive_to_md(value, lvl + 1) + "\n"
        elif isinstance(data, list):
            for i, item in enumerate(data, 1):
                md += f"{'#' * lvl} Item {i}\n\n"
                md += recursive_to_md(item, lvl + 1) + "\n"
        else:
            md += f"{data}\n"
        return md.strip()
        
    return recursive_to_md(data, level)

# docs might be empty if there are no JSON files
markdown_doc = [chunk_to_markdown(doc.page_content) for doc in docs]

encoding = tiktoken.encoding_for_model("gpt-4")

def count_tokens(text):
    """Counts the number of tokens in a given text using the model's tokenizer."""
    return len(encoding.encode(text))

def parse_markdown_tree(md_text):
    lines = md_text.split("\n")
    root = {"level": 0, "heading": None, "content": [], "children": []}
    stack = [root]
    for line in lines:
        m = re.match(r'^(#+)\s+(.*)', line)
        if m:
            level = len(m.group(1))
            heading = m.group(2)
            while stack and stack[-1]['level'] >= level:
                stack.pop()
            node = {"level": level, "heading": heading, "content": [], "children": []}
            stack[-1]['children'].append(node)
            stack.append(node)
        else:
            stack[-1]['content'].append(line)
    return root

def get_heading_context(node, parent_headings=[]):
    """Build the heading context that will be prepended to chunks."""
    headings = [h for h in parent_headings if h]
    if node['heading']:
        headings.append(node['heading'])
    return "\n".join(["#" * (i+1) + " " + h for i, h in enumerate(headings)])

def chunk_node_bottom_up(node, max_tokens=240, parent_headings=[]):
    chunks = []
    for child in node['children']:
        child_chunks = chunk_node_bottom_up(
            child,
            max_tokens=max_tokens,
            parent_headings=parent_headings + ([node['heading']] if node['heading'] else [])
        )
        chunks.extend(child_chunks)
        
    content_text = "\n\n".join(node['content']).strip()
    if not content_text:
        return chunks
        
    heading_context = get_heading_context(node, parent_headings)
    heading_tokens = count_tokens(heading_context) if heading_context else 0
    paragraphs = [p for p in content_text.split("\n\n") if p.strip()]
    
    current_chunk = []
    current_tokens = heading_tokens 
    
    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            full_chunk = f"{heading_context}\n\n{chunk_content}" if heading_context else chunk_content
            chunks.append(full_chunk.strip())
            current_chunk = []
            current_tokens = heading_tokens
                
        if heading_tokens + para_tokens > max_tokens:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent_tokens = count_tokens(sent)
                if current_tokens + sent_tokens > max_tokens and current_chunk:
                    chunk_content = "\n\n".join(current_chunk)
                    full_chunk = f"{heading_context}\n\n{chunk_content}" if heading_context else chunk_content
                    chunks.append(full_chunk.strip())
                    current_chunk = [sent]
                    current_tokens = heading_tokens + sent_tokens
                else:
                    current_chunk.append(sent)
                    current_tokens += sent_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens
            
    if current_chunk:
        chunk_content = "\n\n".join(current_chunk)
        full_chunk = f"{heading_context}\n\n{chunk_content}" if heading_context else chunk_content
        chunks.append(full_chunk.strip())
        
    return chunks

def chunk_markdown_document(md_text, max_tokens=240):
    tree = parse_markdown_tree(md_text)
    return chunk_node_bottom_up(tree, max_tokens=max_tokens)

markdown_chunks_array = []
for doc in markdown_doc:
    text = str(doc)
    tokens = count_tokens(text)
        
    if tokens <= 240:
        markdown_chunks_array.append([text])
    else:
        chunks = chunk_markdown_document(text, max_tokens=240)
        markdown_chunks_array.append(chunks)

markdown_documents = [chunk for doc_chunks in markdown_chunks_array for chunk in doc_chunks]
