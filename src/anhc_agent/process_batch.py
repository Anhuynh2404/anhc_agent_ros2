import json
import os
import math

with open('.understand-anything/tmp/ua-file-extract-results-4.json', 'r') as f:
    data = json.load(f)

with open('.understand-anything/tmp/ua-file-analyzer-input-4.json', 'r') as f:
    input_data = json.load(f)

batch_import_data = input_data.get('batchImportData', {})
project_root = input_data.get('projectRoot', '')

nodes = []
edges = []

def add_node(node):
    nodes.append(node)

def add_edge(edge):
    edges.append(edge)

# Default types based on fileCategory
# fileCategory mapping
category_to_type = {
    'code': 'file',
    'config': 'config',
    'docs': 'document',
    'infra': 'service',  # will adjust below
    'data': 'table',     # will adjust below
    'script': 'file',
    'markup': 'file'
}

for file_result in data.get('results', []):
    path = file_result['path']
    lang = file_result['language']
    cat = file_result['fileCategory']
    
    # Node type logic
    node_type = category_to_type.get(cat, 'file')
    if cat == 'infra':
        if path.startswith('.github/workflows') or path.startswith('.gitlab-ci') or 'Jenkinsfile' in path:
            node_type = 'pipeline'
        elif path.endswith('.tf') or path.endswith('.tfvars') or 'CloudFormation' in path or 'Vagrantfile' in path:
            node_type = 'resource'
        else:
            node_type = 'service'
    elif cat == 'data':
        if path.endswith('.sql'):
            node_type = 'table'
        elif path.endswith('.graphql') or path.endswith('.proto') or path.endswith('.prisma'):
            node_type = 'schema'
        elif 'openapi' in path.lower() or 'swagger' in path.lower():
            node_type = 'endpoint'
        else:
            node_type = 'schema'

    # special rules
    if cat == 'code' and path.endswith('.ipynb'):
        node_type = 'file'

    metrics = file_result.get('metrics', {})
    lines = file_result.get('nonEmptyLines', 0)
    if lines < 50:
        complexity = 'simple'
    elif lines <= 200:
        complexity = 'moderate'
    else:
        complexity = 'complex'

    # Summary and tags
    summary = ""
    tags = []
    
    name = os.path.basename(path)
    
    if path == ".understand-anything/.understandignore":
        summary = "Configuration file specifying paths and patterns to ignore during project analysis."
        tags = ["configuration", "ignore-list"]
        node_type = "config"
    elif path == "anhcape_architecture_viz.ipynb":
        summary = "Jupyter Notebook for visualizing the architecture and performance metrics of the ANHCape agent."
        tags = ["visualization", "jupyter-notebook", "architecture"]
    elif path == "anhcape_architecture.py":
        summary = "Defines the core neural network architecture for the ANHCape agent, including actor and critic networks."
        tags = ["neural-network", "architecture", "data-model"]
    elif path == "CMakeLists.txt":
        summary = "CMake build configuration for the anhc_agent project."
        tags = ["build-system", "configuration"]
    elif path == "evaluation_notebook.ipynb":
        summary = "Jupyter Notebook for evaluating the trained ANHCape agent's performance in simulation."
        tags = ["evaluation", "jupyter-notebook", "test"]
    elif path == "launch/test_anhcape.launch.py":
        summary = "ROS 2 launch file for running tests and evaluating the ANHCape agent."
        tags = ["launch-file", "test", "configuration"]
    elif path == "package.xml":
        summary = "ROS 2 package configuration detailing dependencies and package metadata."
        tags = ["package-config", "configuration"]
    elif path == "scripts/__init__.py":
        summary = "Initialization file for the scripts module."
        tags = ["entry-point", "barrel"]
    elif path == "scripts/environment/__init__.py":
        summary = "Initialization file for the environment module."
        tags = ["entry-point", "barrel"]
    elif path == "scripts/environment/environment_interface.py":
        summary = "Abstract interface defining the required methods for RL environment wrappers."
        tags = ["interface", "type-definition", "data-model"]
    elif path == "scripts/environment/environment.py":
        summary = "Implementation of the ROS 2 based reinforcement learning environment for the ANHCape agent."
        tags = ["environment", "simulation", "service"]
    elif path == "scripts/policy/__init__.py":
        summary = "Initialization file for the policy module."
        tags = ["entry-point", "barrel"]
    elif path == "scripts/utils/__init__.py":
        summary = "Initialization file for the utils module."
        tags = ["entry-point", "barrel"]
    elif path == "scripts/utils/buffer.py":
        summary = "Implementation of replay buffers (e.g., PrioritizedReplayBuffer) for off-policy reinforcement learning."
        tags = ["utility", "data-structure", "data-model"]
    elif path == "scripts/utils/plot_reward.py":
        summary = "Utility script for plotting training rewards and performance metrics."
        tags = ["utility", "visualization", "script"]
    elif path == "scripts/utils/point_cloud2.py":
        summary = "Utility functions for working with ROS 2 PointCloud2 messages, converting to and from numpy arrays."
        tags = ["utility", "data-processing", "serialization"]
        
    if not tags:
        tags = ["file"]

    file_node_id = f"{node_type}:{path}"
    
    add_node({
        "id": file_node_id,
        "type": node_type,
        "name": name,
        "filePath": path,
        "summary": summary,
        "tags": tags,
        "complexity": complexity
    })

    # imports
    imports = batch_import_data.get(path, [])
    for imp in imports:
        add_edge({
            "source": file_node_id,
            "target": f"file:{imp}",
            "type": "imports",
            "direction": "forward",
            "weight": 0.7
        })

    # functions and classes
    for func in file_result.get('functions', []):
        func_lines = func['endLine'] - func['startLine']
        is_exported = any(e['name'] == func['name'] for e in file_result.get('exports', []))
        if func_lines >= 10 or is_exported:
            func_id = f"function:{path}:{func['name']}"
            add_node({
                "id": func_id,
                "type": "function",
                "name": func['name'],
                "filePath": path,
                "summary": f"Function {func['name']} in {name}.",
                "tags": ["function", "utility"] if "util" in path else ["function", "component"],
                "complexity": "simple" if func_lines < 20 else ("moderate" if func_lines < 50 else "complex"),
                "lineRange": [func['startLine'], func['endLine']]
            })
            add_edge({
                "source": file_node_id,
                "target": func_id,
                "type": "contains",
                "direction": "forward",
                "weight": 1.0
            })
            if is_exported:
                add_edge({
                    "source": file_node_id,
                    "target": func_id,
                    "type": "exports",
                    "direction": "forward",
                    "weight": 0.8
                })

    for cls in file_result.get('classes', []):
        cls_lines = cls['endLine'] - cls['startLine']
        methods_count = len(cls.get('methods', []))
        is_exported = any(e['name'] == cls['name'] for e in file_result.get('exports', []))
        if cls_lines >= 20 or methods_count >= 2 or is_exported:
            cls_id = f"class:{path}:{cls['name']}"
            add_node({
                "id": cls_id,
                "type": "class",
                "name": cls['name'],
                "filePath": path,
                "summary": f"Class {cls['name']} in {name}.",
                "tags": ["class", "data-model"] if "model" in path else ["class", "component"],
                "complexity": "simple" if cls_lines < 50 else ("moderate" if cls_lines < 150 else "complex"),
                "lineRange": [cls['startLine'], cls['endLine']]
            })
            add_edge({
                "source": file_node_id,
                "target": cls_id,
                "type": "contains",
                "direction": "forward",
                "weight": 1.0
            })
            if is_exported:
                add_edge({
                    "source": file_node_id,
                    "target": cls_id,
                    "type": "exports",
                    "direction": "forward",
                    "weight": 0.8
                })

# Missing files that script skipped?
extracted_paths = set(r['path'] for r in data.get('results', []))
for f in input_data.get('batchFiles', []):
    if f['path'] not in extracted_paths:
        path = f['path']
        name = os.path.basename(path)
        cat = f['fileCategory']
        node_type = category_to_type.get(cat, 'file')
        if cat == 'infra':
            node_type = 'service'
        elif cat == 'data':
            node_type = 'table'
        file_node_id = f"{node_type}:{path}"
        
        complexity = 'simple'
        if f['sizeLines'] > 200: complexity = 'complex'
        elif f['sizeLines'] > 50: complexity = 'moderate'
        
        # we can just use our mappings
        summary = f"{name} file."
        tags = ["file"]
        if path == "scripts/__init__.py":
            summary = "Initialization file for the scripts module."
            tags = ["entry-point", "barrel"]
        elif path == "scripts/environment/__init__.py":
            summary = "Initialization file for the environment module."
            tags = ["entry-point", "barrel"]
        elif path == "scripts/policy/__init__.py":
            summary = "Initialization file for the policy module."
            tags = ["entry-point", "barrel"]
        elif path == "scripts/utils/__init__.py":
            summary = "Initialization file for the utils module."
            tags = ["entry-point", "barrel"]

        add_node({
            "id": file_node_id,
            "type": node_type,
            "name": name,
            "filePath": path,
            "summary": summary,
            "tags": tags,
            "complexity": complexity
        })
        # imports
        imports = batch_import_data.get(path, [])
        for imp in imports:
            add_edge({
                "source": file_node_id,
                "target": f"file:{imp}",
                "type": "imports",
                "direction": "forward",
                "weight": 0.7
            })

# manual edges for CMakeLists and package.xml and others
edges.extend([
    {
        "source": "document:CMakeLists.txt",
        "target": "config:package.xml",
        "type": "related",
        "direction": "forward",
        "weight": 0.5
    },
    {
        "source": "config:package.xml",
        "target": "document:CMakeLists.txt",
        "type": "related",
        "direction": "forward",
        "weight": 0.5
    },
    {
        "source": "file:anhcape_architecture_viz.ipynb",
        "target": "file:anhcape_architecture.py",
        "type": "related",
        "direction": "forward",
        "weight": 0.5
    },
    {
        "source": "file:evaluation_notebook.ipynb",
        "target": "file:scripts/environment/environment.py",
        "type": "related",
        "direction": "forward",
        "weight": 0.5
    },
    {
        "source": "file:evaluation_notebook.ipynb",
        "target": "file:anhcape_architecture.py",
        "type": "related",
        "direction": "forward",
        "weight": 0.5
    },
    {
        "source": "file:launch/test_anhcape.launch.py",
        "target": "file:scripts/environment/environment.py",
        "type": "triggers",
        "direction": "forward",
        "weight": 0.6
    }
])

out_data = {
    "nodes": nodes,
    "edges": edges
}

nodeCount = len(nodes)
edgeCount = len(edges)

if nodeCount <= 60 and edgeCount <= 120:
    with open('.understand-anything/intermediate/batch-4.json', 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"Wrote single part. Nodes: {nodeCount}, Edges: {edgeCount}")
else:
    parts = math.ceil(max(nodeCount / 60, edgeCount / 120))
    print(f"Splitting into {parts} parts. Nodes: {nodeCount}, Edges: {edgeCount}")
    # group files
    file_paths = sorted(list(set([n.get('filePath') for n in nodes if n.get('filePath')])))
    chunk_size = math.ceil(len(file_paths) / parts)
    
    for i in range(parts):
        chunk_files = set(file_paths[i*chunk_size : (i+1)*chunk_size])
        
        part_nodes = [n for n in nodes if n.get('filePath') in chunk_files]
        part_node_ids = set(n['id'] for n in part_nodes)
        
        part_edges = [e for e in edges if e['source'] in part_node_ids]
        
        with open(f'.understand-anything/intermediate/batch-4-part-{i+1}.json', 'w') as f:
            json.dump({"nodes": part_nodes, "edges": part_edges}, f, indent=2)
        print(f"Wrote part {i+1}. Nodes: {len(part_nodes)}, Edges: {len(part_edges)}")

