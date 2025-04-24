from flask import Flask, request, jsonify, render_template, send_from_directory
import nltk
import torch
import json
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive mode
import matplotlib.pyplot as plt
from nltk.tree import Tree
import os
import subprocess
from parser_trainer import SimpleParser

app = Flask(__name__)

# Load the trained model and mappings
def load_model():
    with open('static/word_to_idx.json', 'r') as f:
        word_to_idx = json.load(f)
    with open('static/tag_to_idx.json', 'r') as f:
        tag_to_idx = json.load(f)
    
    # Model parameters
    vocab_size = len(word_to_idx)
    embedding_dim = 100
    hidden_dim = 128
    output_dim = len(tag_to_idx)
    
    model = SimpleParser(vocab_size, embedding_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load('static/parser_model.pth'))
    model.eval()
    
    return model, word_to_idx, tag_to_idx

# Grammar for parsing
GRAMMAR = {
    'S': [['NP', 'VP'], ['DT', 'S']],
    'NP': [['DT', 'NN'], ['PRP'], ['NNP'], ['DT', 'S']],
    'VP': [['VBD', 'NP'], ['VBD', 'PP'], ['VBZ', 'NP'], ['VBZ', 'PP']],
    'PP': [['IN', 'NP']],
}

class TreeNode:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children if children else []
        self.word = None
        self.tag = None
    
    def __str__(self):
        if self.word:
            return f"{self.word}({self.tag})"
        return self.label

def top_down_parse(words, tags):
    def expand(symbol, pos):
        if pos >= len(words):
            return None, pos
        
        # If symbol is a terminal (POS tag)
        if symbol in ['DT', 'NN', 'VBD', 'IN', 'PRP', 'NNP', 'VBZ']:
            if pos < len(tags) and tags[pos].startswith(symbol):
                node = TreeNode(symbol)
                node.word = words[pos]
                node.tag = tags[pos]
                return node, pos + 1
            return None, pos
        
        # If symbol is non-terminal
        if symbol in GRAMMAR:
            for production in GRAMMAR[symbol]:
                node = TreeNode(symbol)
                current_pos = pos
                all_matched = True
                
                for sym in production:
                    child_node, new_pos = expand(sym, current_pos)
                    if child_node is None:
                        all_matched = False
                        break
                    node.children.append(child_node)
                    current_pos = new_pos
                
                if all_matched:
                    return node, current_pos
        
        return None, pos
    
    root, _ = expand('S', 0)
    return root

def bottom_up_parse(words, tags):
    # Initialize leaf nodes
    nodes = []
    for word, tag in zip(words, tags):
        node = TreeNode(tag)
        node.word = word
        node.tag = tag
        nodes.append(node)
    
    print("\nBottom-up parsing debug:")
    print("Initial nodes:", [f"{n.word}({n.label})" for n in nodes])
    
    # Keep reducing until no more reductions possible or only one node left
    iteration = 0
    while len(nodes) > 1:
        print(f"\nIteration {iteration}:")
        print("Current nodes:", [f"{n.word}({n.label})" if n.word else n.label for n in nodes])
        reduced = False
        # Try each possible starting position
        for i in range(len(nodes)):
            if reduced:
                break
            # Try each grammar rule
            for lhs, rhs_list in GRAMMAR.items():
                if reduced:
                    break
                for rhs in rhs_list:
                    # Check if we have enough nodes left to match this rule
                    if i + len(rhs) <= len(nodes):
                        # Get the sequence of node labels
                        sequence = [n.label for n in nodes[i:i+len(rhs)]]
                        print(f"  Position {i}: Trying to match {sequence} with rule {lhs} -> {rhs}")
                        # Check if sequence exactly matches the production rule
                        if sequence == rhs:
                            print(f"  Found match! Reducing {sequence} to {lhs}")
                            # Create new node with matched nodes as children
                            new_node = TreeNode(lhs)
                            new_node.children = nodes[i:i+len(rhs)]
                            # Replace matched nodes with new node
                            nodes[i:i+len(rhs)] = [new_node]
                            reduced = True
                            break
        
        # If no reduction was possible, we're done
        if not reduced:
            print("\nNo more reductions possible")
            break
        iteration += 1
    
    print("\nFinal nodes:", [f"{n.word}({n.label})" if n.word else n.label for n in nodes])
    
    # Check if we got a valid parse (single node with label 'S')
    if len(nodes) == 1 and nodes[0].label == 'S':
        print("Successfully parsed to S!")
        return nodes[0]
    print("Failed to parse to S")
    return None

def create_visualization(tree, parser_type):
    if not tree:
        return None
        
    try:
        print(f"\nVisualization debug for {parser_type}:")
        # Create DOT file content
        dot_content = [
            f"// {parser_type} Parse Tree",
            "digraph {",
            "\trankdir=TB"
        ]
        
        def add_to_dot(node, parent_id=None):
            node_id = str(id(node))
            # Create node label
            if node.word:
                label = f"{node.word}\\n({node.tag})"
            else:
                label = node.label
                
            # Add node
            dot_content.append(f'\t{node_id} [label="{label}"]')
            
            # Add edge if there's a parent
            if parent_id:
                dot_content.append(f'\t{parent_id} -> {node_id}')
                
            # Process children
            for child in node.children:
                add_to_dot(child, node_id)
        
        # Build the DOT content
        add_to_dot(tree)
        dot_content.append("}")
        
        # Write DOT file
        dot_filename = os.path.join('static', f'{parser_type.lower().replace("-", "_")}_tree')
        with open(dot_filename, 'w') as f:
            f.write('\n'.join(dot_content))
        
        print(f"DOT file written to: {dot_filename}")
        print(f"DOT file exists: {os.path.exists(dot_filename)}")
        print(f"DOT file content:\n{open(dot_filename).read()}")
        
        # Convert DOT to PNG using Graphviz
        png_filename = dot_filename + '.png'
        dot_path = r"C:\Program Files\Graphviz\bin\dot.exe"
        
        print(f"\nGraphviz debug:")
        print(f"dot_path exists: {os.path.exists(dot_path)}")
        print(f"Running command: {dot_path} -Tpng {dot_filename} -o {png_filename}")
        
        # Run dot command with full path
        result = subprocess.run(
            [dot_path, '-Tpng', dot_filename, '-o', png_filename],
            check=True,
            capture_output=True,
            text=True
        )
        
        print(f"Command output: {result.stdout}")
        if result.stderr:
            print(f"Command error: {result.stderr}")
        
        print(f"PNG file written to: {png_filename}")
        print(f"PNG file exists: {os.path.exists(png_filename)}")
        
        # Return the filename without path
        return f'{parser_type.lower().replace("-", "_")}_tree.png'
        
    except Exception as e:
        print(f"Error generating visualization: {str(e)}")
        print(f"Full error details: {e.__class__.__name__}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/parse', methods=['POST'])
def parse():
    try:
        sentence = request.json['sentence']
        parse_type = request.json['type']
        
        print("\n" + "="*80)
        print(f"PARSING REQUEST: {parse_type}")
        print(f"Sentence: {sentence}")
        print("="*80)
        
        # Get POS tags using the trained model
        model, word_to_idx, tag_to_idx = load_model()
        words = sentence.split()
        word_indices = [word_to_idx.get(w.lower(), word_to_idx['<unk>']) for w in words]
        word_tensor = torch.LongTensor([word_indices])
        
        with torch.no_grad():
            outputs = model(word_tensor)
            predictions = outputs.argmax(dim=2)[0]
        
        idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
        tags = [idx_to_tag[idx.item()] for idx in predictions]
        
        # Map model's POS tags to our grammar's tags
        tag_mapping = {
            'NN': 'NN',
            'NNS': 'NN',
            'NNP': 'NNP',
            'NNPS': 'NNP',
            'DT': 'DT',
            'VB': 'VBZ',
            'VBZ': 'VBZ',
            'VBD': 'VBD',
            'VBP': 'VBZ',
            'VBG': 'VBZ',
            'IN': 'IN',
            'PRP': 'PRP'
        }
        
        mapped_tags = [tag_mapping.get(tag, tag) for tag in tags]
        print("\nPOS Tagging:")
        print(f"Original tags: {tags}")
        print(f"Mapped tags: {mapped_tags}")
        print("\nGrammar Rules:")
        for lhs, rhs_list in GRAMMAR.items():
            print(f"{lhs} -> {' | '.join([' '.join(rhs) for rhs in rhs_list])}")
        
        # Parse using selected strategy
        print(f"\nStarting {parse_type} parsing...")
        if parse_type == 'top-down':
            tree = top_down_parse(words, mapped_tags)
        else:
            tree = bottom_up_parse(words, mapped_tags)
        
        if tree:
            print("\nParsing successful!")
            print("Generating visualization...")
            image_path = create_visualization(tree, parse_type)
            if image_path:
                print(f"Visualization saved as: {image_path}")
                return jsonify({
                    'success': True,
                    'image': f'/static/{image_path}',
                    'tags': tags
                })
            else:
                print("Failed to generate visualization")
        else:
            print("\nParsing failed - could not build a valid parse tree")
        
        return jsonify({
            'success': False,
            'error': 'Could not parse the sentence'
        })
        
    except Exception as e:
        print(f"\nError in parse endpoint: {str(e)}")
        print("Full error details:")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Ensure static directory exists
    os.makedirs('static', exist_ok=True)
    app.run(debug=True) 