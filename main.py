import io
import os
import re
import json
import math
import colorsys
from collections import deque
from typing import List, Dict, Any, Tuple
import pprint
import mimetypes

import numpy as np
from PIL import Image
import pymupdf
import ezdxf
from ezdxf.addons.drawing import Frontend, RenderContext, layout
from ezdxf.addons.drawing import pymupdf
from ezdxf.math import Vec2
from ezdxf.math import BoundingBox
from scipy.spatial import cKDTree, ConvexHull
from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import rotate, translate
from fastapi import HTTPException

from call_claude import call_claude

Image.MAX_IMAGE_PIXELS = None

def extract_entity_vertices(entity):
    """Extracts all significant coordinate points from a DXF entity."""
    points = []
    dxftype = entity.dxftype()
    
    if dxftype == 'LINE':
        points.append((entity.dxf.start.x, entity.dxf.start.y))
        points.append((entity.dxf.end.x, entity.dxf.end.y))
        
    elif dxftype in ('LWPOLYLINE', 'POLYLINE'):
        for vertex in entity.get_points():
            points.append((vertex[0], vertex[1]))
            
    elif dxftype in ('CIRCLE', 'ARC'):
        cx, cy = entity.dxf.center.x, entity.dxf.center.y
        r = entity.dxf.radius
        points.append((cx, cy))
        points.append((cx + r, cy))
        points.append((cx - r, cy))
        points.append((cx, cy + r))
        points.append((cx, cy - r))
        
    return points

async def group_by_vertex_chain(dxf_path, layer_name="ANTENNAS DISHES", max_link_distance=100.0):
    try:
        doc = ezdxf.readfile(dxf_path)
    except IOError:
        raise HTTPException(status_code=400, detail="Cannot open or find the DXF file.")
    except ezdxf.DXFStructureError:
        raise HTTPException(status_code=400, detail="Invalid or corrupted DXF structure.")
        
    msp = doc.modelspace()
    layer_entities = list(msp.query(f'*[layer=="{layer_name}"]'))
    
    if not layer_entities:
        return {"status": "success", "message": f"No entities found on layer '{layer_name}'", "clusters": []}
        
    all_points = []
    point_to_entity_idx = []
    
    for ent_idx, entity in enumerate(layer_entities):
        vertices = extract_entity_vertices(entity)
        for pt in vertices:
            all_points.append(pt)
            point_to_entity_idx.append(ent_idx)
            
    if not all_points:
        raise HTTPException(status_code=422, detail="No processable geometric vertices found on this layer.")
        
    X = np.array(all_points)
    point_to_entity_idx = np.array(point_to_entity_idx)
    
    tree = cKDTree(X)
    num_entities = len(layer_entities)
    adjacency_list = {i: set() for i in range(num_entities)}
    
    pairs = tree.query_pairs(r=max_link_distance)
    for p1, p2 in pairs:
        ent1 = point_to_entity_idx[p1]
        ent2 = point_to_entity_idx[p2]
        if ent1 != ent2:
            adjacency_list[ent1].add(ent2)
            adjacency_list[ent2].add(ent1)
            
    visited = [False] * num_entities
    clusters = []
    
    for ent_idx in range(num_entities):
        if visited[ent_idx]:
            continue
            
        queue = deque([ent_idx])
        visited[ent_idx] = True
        cluster_entities = []
        
        while queue:
            curr = queue.popleft()
            entity = layer_entities[curr]
            cluster_entities.append(entity)
            
            for neighbor in adjacency_list[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    
        clusters.append(cluster_entities)

    return {"doc": doc, "clusters": clusters}

async def label_dxf_clusters(
    file_path: str, 
    output_suffix: str = "_labeled", 
    text_height: float = 50.0, 
    offset_distance: float = 20.0
) -> dict:
    """
    Reads the DXF file, groups components, overlays 'antenna_X' labels,
    saves the modified CAD file, and returns serializable cluster metadata.
    """
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Source DXF file not found.")

    data = await group_by_vertex_chain(file_path)
    doc = data["doc"]
    clusters = data["clusters"]
    
    msp = doc.modelspace()
    
    label_layer = "ANTENNA_LABELS"
    if label_layer not in doc.layers:
        doc.layers.new(name=label_layer, dxfattribs={'color': 2}) 

    serialized_clusters = {}

    for idx, cluster in enumerate(clusters):
        cluster_color = (idx % 6) + 1
        cluster_points = []
        entities_metadata = []
        
        for entity in cluster:
            raw_vertices = extract_entity_vertices(entity)
            cluster_points.extend(raw_vertices)
            entity.dxf.color = cluster_color
            
            entities_metadata.append({
                "id": str(entity.dxf.handle),       
                "type": str(entity.dxftype()),       
                "vertices": raw_vertices             
            })
            
        if not cluster_points:
            continue
            
        arr = np.unique(np.array(cluster_points)[:, :2], axis=0)
        
        if len(arr) >= 3:
            hull = ConvexHull(arr)
            boundary_vertices = arr[hull.vertices].tolist()
        else:
            boundary_vertices = arr.tolist()

        min_x, min_y = np.min(arr, axis=0)
        max_x, max_y = np.max(arr, axis=0)
        
        label_x = min_x
        label_y = max_y + offset_distance
        
        cluster_key = f"antenna_{idx}"
        
        msp.add_text(
            text=cluster_key,
            dxfattribs={
                'layer': label_layer,
                'color': cluster_color,
                'height': text_height,
                'insert': (label_x, label_y)
            }
        )
        
        center_x, center_y = np.mean(arr, axis=0)
        
        serialized_clusters[cluster_key] = {
            "center": [float(center_x), float(center_y)],
            "bounding_box": boundary_vertices, 
            "entities": entities_metadata
        }

    base, ext = os.path.splitext(file_path)
    output_path = f"{base}{output_suffix}{ext}"
    doc.saveas(output_path)
    
    return {
        "status": "success",
        "output_path": output_path,
        "clusters": serialized_clusters
    }

async def label_and_snapshot_dxf(
    dxf_path: str, 
    output_img_name: str = "dxf_snap", 
    max_edge_pixels: int = 2048  
):
    if not os.path.exists(dxf_path):
        raise HTTPException(status_code=404, detail="Target DXF file not found.")
        
    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        layer_name = "ANTENNAS DISHES"
        layer_entities = list(msp.query(f'*[layer=="{layer_name}"]'))
        if not layer_entities:
            layer_entities = list(msp)
            
        if not layer_entities:
            raise HTTPException(status_code=422, detail="No visible geometry found to render.")
            
        all_scene_points = []
        for entity in layer_entities:
            entity_pts = extract_entity_vertices(entity)
            if entity_pts:
                all_scene_points.extend(entity_pts)
                
        scene_arr = np.array(all_scene_points)
        s_min_x, s_min_y = np.min(scene_arr, axis=0)
        s_max_x, s_max_y = np.max(scene_arr, axis=0)
        
        margin_y = 400.0  
        margin_x = 800.0  

        width = (s_max_x - s_min_x) + (margin_x * 2)
        height = (s_max_y - s_min_y) + (margin_y * 2)
        
        render_box = BoundingBox([
            (s_min_x - margin_x, s_min_y - margin_y), 
            (s_max_x + margin_x, s_max_y + margin_y)
        ])
        
        max_drawing_units = max(width, height)
        calculated_dpi = int((max_edge_pixels / max_drawing_units) * 72)
        calculated_dpi = max(min(calculated_dpi, 300), 45)
        
        ctx = RenderContext(doc)
        backend = pymupdf.PyMuPdfBackend()
        Frontend(ctx, backend).draw_layout(msp)
        
        ppm_bytes = backend.get_pixmap_bytes(
            page=layout.Page(0, 0),
            fmt="ppm",
            dpi=calculated_dpi,
            render_box=render_box
        )

        dxf_filename = os.path.basename(dxf_path)  
        dxf_name = os.path.splitext(dxf_filename)[0]
        output_path = f"{output_img_name}/{dxf_name}.png"
        
        with io.BytesIO(ppm_bytes) as stream:
            with Image.open(stream, formats=["ppm"]) as img:
                img.save(output_path, format="PNG")
                
        return {
            "status": "success",
            "saved_snapshot": output_path,
            "rendered_dpi": calculated_dpi
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate snapshot: {str(e)}")

async def get_antenna(img_path: str, snapshot_path: str, input_params: str):
    try:
        prompt = """
            You will be given two images and an input text. Image 1 is a labeled cluster image showing antennas with labels such as antenna_1, antenna_2, antenna_3, and so on marked on a steelwork structure. Image 2 is a steelwork image showing the physical antenna installation. The input text contains an antenna name and three azimuth values that together cover 360 degrees of directional coverage.

Your job is to follow these steps:

STEP 1 - Match the input text to the labeled cluster image. Read the antenna name and the three azimuth values from the input text. Look at Image 1 carefully. Identify which antenna in the labeled image matches the antenna name and corresponds to each azimuth direction. Use the azimuth values as spatial cues to determine which labeled antenna is facing or aligned with that direction.

STEP 2 - Map the matched antenna to the steelwork image. Using the match found in Step 1, locate the corresponding antenna label in Image 2. Confirm that the label region in the steelwork image contains the main radiating front face of the antenna and not just mounting brackets, connecting rods, clamps, or support poles.

STEP 3 - Validate and apply fallback if needed. If the matched antenna label in the steelwork image does not contain the main front antenna panel and only shows brackets, clamps, rods, or other connecting hardware, then do the following. First, try the next closest label in the same azimuth region. If that also fails, move to one of the other two azimuth values and repeat Steps 1 and 2 for that direction. Keep trying until you find a label that clearly contains the actual front face of the antenna.

Important rules:

A label is only valid if it visually contains the main front radiating panel of the antenna.
Brackets, clamps, rods, or rear and side views alone are not a valid match.
Always prefer the label that is most closely aligned with the stated azimuth direction.
If no valid match is found across all three azimuths, your reasoning must clearly state that no valid match was found and explain why.

Output Format:
Output your final answer strictly as a JSON object matching the schema below. Do not include any conversational text outside the JSON structure.

{
    "status": "Success",
    "matched_index": "antenna_3",
    "reasoning": "small brief reasoning for selection, including whether a fallback was used and confirmation of the front face."
}
(Note: Use "Success" for status if a valid front face is found, or "Failure" if no valid match is found. If failed, matched_index should be null.)
         """

        filled_prompt = f"""
            {prompt} 

            Input Param:
            {input_params}
          """

        for path in [img_path, snapshot_path]:
            if not os.path.exists(path):
                return {"status": "failure", "reason": f"File not found for antenna cluster identification at {path}"}

        formatted_files_data = []
        
        def get_content_type(filename):
            ext = os.path.splitext(filename)[1].lower()
            return "image/jpeg" if ext in ['.jpg', '.jpeg'] else "image/png"

        with open(img_path, "rb") as f:
            img_bytes = f.read()
        img_filename = os.path.basename(img_path)
        formatted_files_data.append((img_filename, img_bytes, get_content_type(img_filename)))

        with open(snapshot_path, "rb") as f:
            snap_bytes = f.read()
        snap_filename = os.path.basename(snapshot_path)
        formatted_files_data.append((snap_filename, snap_bytes, get_content_type(snap_filename)))

        response_text = await call_claude(filled_prompt, formatted_files_data)

        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
            
        clean_text = clean_text.strip()
        parsed_json = json.loads(clean_text)

        return parsed_json

    except Exception as e:
        return {"status": "failure", "reason": str(e)}

# async def process_antenna_radiation(
#     file_path: str,
#     clusters_dict: dict,  
#     matched_index_str: str,
#     azimuth_deg: float,
#     radiation_len: float = 2000.0
# ) -> str:
#     """
#     Opens the existing DXF file, locates the target cluster, calculates its 
#     front face boundary, backs away 600mm from that face, and fires radiation rays.
#     """
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Target DXF file not found at: {file_path}")

#     if matched_index_str not in clusters_dict:
#         return file_path

#     target_antenna_data = clusters_dict[matched_index_str]
#     center_x, center_y = target_antenna_data["center"]
    
#     bbox_vertices = np.array(target_antenna_data["bounding_box"])
#     min_x, min_y = np.min(bbox_vertices, axis=0)
#     max_x, max_y = np.max(bbox_vertices, axis=0)
    
#     antenna_width = max_x - min_x
#     antenna_height = max_y - min_y
    
#     math_theta = math.radians(90.0 - azimuth_deg)
    
#     cos_t = math.cos(math_theta)
#     sin_t = math.sin(math_theta)
    
#     dist_x = abs((antenna_width / 2.0) / cos_t) if abs(cos_t) > 1e-5 else float('inf')
#     dist_y = abs((antenna_height / 2.0) / sin_t) if abs(sin_t) > 1e-5 else float('inf')
    
#     distance_to_front_face = min(dist_x, dist_y)
#     net_shift = distance_to_front_face - 300.0
    
#     source_x = center_x + net_shift * cos_t
#     source_y = center_y + net_shift * sin_t
#     source_pt = (source_x, source_y)
    
#     doc = ezdxf.readfile(file_path)
#     msp = doc.modelspace()
    
#     rad_layer = "ANTENNA_RADIATION"
#     if rad_layer not in doc.layers:
#         doc.layers.new(name=rad_layer, dxfattribs={'color': 1}) 
        
#     angle_plus_60 = math_theta + math.radians(60.0)
#     angle_minus_60 = math_theta - math.radians(60.0)
    
#     ray1_end = (
#         source_x + radiation_len * math.cos(angle_plus_60),
#         source_y + radiation_len * math.sin(angle_plus_60)
#     )
#     ray2_end = (
#         source_x + radiation_len * math.cos(angle_minus_60),
#         source_y + radiation_len * math.sin(angle_minus_60)
#     )
    
#     msp.add_line(start=source_pt, end=ray1_end, dxfattribs={'layer': rad_layer})
#     msp.add_line(start=source_pt, end=ray2_end, dxfattribs={'layer': rad_layer})
    
#     deg_start = math.degrees(angle_minus_60)
#     deg_end = math.degrees(angle_plus_60)
#     msp.add_arc(
#         center=source_pt,
#         radius=radiation_len / 2.0,
#         start_angle=deg_start,
#         end_angle=deg_end,
#         dxfattribs={'layer': rad_layer}
#     )
    
#     doc.saveas(file_path)
#     return file_path, (source_x, source_y)   


async def process_antenna_radiation(
    file_path: str,
    clusters_dict: dict,  
    matched_index_str: str,
    azimuth_deg: float,
    radiation_len: float = 2000.0
) -> str:
    """
    Opens the existing DXF file, locates the target cluster, calculates its 
    front face boundary, backs away 600mm from that face, and fires radiation rays.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target DXF file not found at: {file_path}")

    if matched_index_str not in clusters_dict:
        return file_path

    target_antenna_data = clusters_dict[matched_index_str]
    center_x, center_y = target_antenna_data["center"]
    center_pt = np.array([center_x, center_y])
    
    bbox_vertices = np.array(target_antenna_data["bounding_box"])
    
    math_theta = math.radians(90.0 - azimuth_deg)
    
    cos_t = math.cos(math_theta)
    sin_t = math.sin(math_theta)
    direction = np.array([cos_t, sin_t])
    
    # True front-face distance: project every hull vertex (relative to the
    # cluster's actual center) onto the azimuth direction. The farthest
    # projection IS the front face, regardless of how the cluster's shape
    # is rotated/irregular in the drawing - unlike the old axis-aligned
    # bbox approximation, which could land outside the real face.
    projections = (bbox_vertices - center_pt) @ direction
    distance_to_front_face = float(np.max(projections))
    
    # Walk 300 units back from the front face, straight along the azimuth
    # axis through the center - so the source always stays on the
    # center-line, never skewed toward a corner.
    net_shift = distance_to_front_face - 200.0
    
    source_x = center_x + net_shift * cos_t
    source_y = center_y + net_shift * sin_t
    source_pt = (source_x, source_y)
    
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    
    rad_layer = "ANTENNA_RADIATION"
    if rad_layer not in doc.layers:
        doc.layers.new(name=rad_layer, dxfattribs={'color': 1}) 
        
    angle_plus_60 = math_theta + math.radians(60.0)
    angle_minus_60 = math_theta - math.radians(60.0)
    
    ray1_end = (
        source_x + radiation_len * math.cos(angle_plus_60),
        source_y + radiation_len * math.sin(angle_plus_60)
    )
    ray2_end = (
        source_x + radiation_len * math.cos(angle_minus_60),
        source_y + radiation_len * math.sin(angle_minus_60)
    )
    
    msp.add_line(start=source_pt, end=ray1_end, dxfattribs={'layer': rad_layer})
    msp.add_line(start=source_pt, end=ray2_end, dxfattribs={'layer': rad_layer})
    
    deg_start = math.degrees(angle_minus_60)
    deg_end = math.degrees(angle_plus_60)
    msp.add_arc(
        center=source_pt,
        radius=radiation_len / 2.0,
        start_angle=deg_start,
        end_angle=deg_end,
        dxfattribs={'layer': rad_layer}
    )
    
    doc.saveas(file_path)
    return file_path, (source_x, source_y)   


def get_significant_vertices(
    file_path: str,
    layer_name: str = "STEELWORK",
    min_member_length: float = 300.0,   
    joint_cluster_dist: float = 200.0,  
    min_degree: int = 2,                
) -> list:
    """
    Extracts the true structural joint vertices from the steelwork layer.
    """
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    endpoints: list = []

    for entity in msp.query(f'*[layer=="{layer_name}"]'):
        dt = entity.dxftype()

        if dt == 'LINE':
            s = (entity.dxf.start.x, entity.dxf.start.y)
            e = (entity.dxf.end.x, entity.dxf.end.y)
            if math.hypot(e[0] - s[0], e[1] - s[1]) >= min_member_length:
                endpoints.extend([s, e])

        elif dt in ('LWPOLYLINE', 'POLYLINE'):
            if entity.closed:
                continue 
            pts = [(v[0], v[1]) for v in entity.get_points()]
            if len(pts) < 2:
                continue
            for i in range(len(pts) - 1):
                seg_len = math.hypot(pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1])
                if seg_len >= min_member_length:
                    endpoints.extend([pts[i], pts[i+1]])

    assigned = [False] * len(endpoints)
    clusters = [] 

    for i, pt in enumerate(endpoints):
        if assigned[i]:
            continue
        group = [pt]
        assigned[i] = True
        for j in range(i + 1, len(endpoints)):
            if not assigned[j] and math.hypot(pt[0] - endpoints[j][0], pt[1] - endpoints[j][1]) < joint_cluster_dist:
                group.append(endpoints[j])
                assigned[j] = True
        cx = sum(p[0] for p in group) / len(group)
        cy = sum(p[1] for p in group) / len(group)
        clusters.append((cx, cy, len(group)))

    joints_with_deg = [(cx, cy, deg) for cx, cy, deg in clusters if deg >= min_degree]
    joints_with_deg.sort(key=lambda v: (-v[2], v[1], v[0]))

    return [(cx, cy) for cx, cy, deg in joints_with_deg]

async def label_structural_vertices(file_path: str, layer_name: str = "STEELWORK", text_height: float = 50.0, offset_distance: float = 15.0,dot_radius: float = 20.0 ) -> dict:
    """
    Computes key structural anchors, maps them to sequential P-keys, 
    and writes them to the VERTEX_LABELS layer inside the DXF document.
    """
    vertices = get_significant_vertices(file_path, layer_name=layer_name)

    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    label_layer = "VERTEX_LABELS"
    if label_layer not in doc.layers:
        doc.layers.new(name=label_layer, dxfattribs={'color': 7})  

    n = max(len(vertices), 1)
    GOLDEN_RATIO = 0.618033988749895  

    vertex_map = {}
    for idx, (vx, vy) in enumerate(vertices):
        point_key = f"P{idx + 1}"
        vertex_map[point_key] = [float(vx), float(vy)]

        hue = (idx * GOLDEN_RATIO) % 1.0
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.90, 1.0)]
        true_color_val = (r << 16) | (g << 8) | b  

        color_attrs = {
            'layer': label_layer,
            'true_color': true_color_val
        }

        r_half = dot_radius / 2.0
        dot = msp.add_lwpolyline(
            [(vx - r_half, vy, 1.0), (vx + r_half, vy, 1.0)],
            format="xyb",
            close=True,
            dxfattribs=color_attrs
        )
        dot.dxf.const_width = dot_radius

        msp.add_text(
            text=point_key,
            dxfattribs={
                **color_attrs,
                'height': text_height,
                'insert': (vx + offset_distance, vy + offset_distance)
            }
        )

    doc.saveas(file_path)
    return vertex_map

async def snapshot_steelwork_and_labels(
    dxf_path: str, 
    output_dir: str = "dxf_snap", 
    max_edge_pixels: int = 2048
) -> str:
    """
    Renders a high-contrast PNG snippet displaying ONLY the STEELWORK 
    and the new VERTEX_LABELS layer.
    """
    from ezdxf.addons.drawing.config import Configuration, BackgroundPolicy

    if not os.path.exists(dxf_path):
        raise HTTPException(status_code=404, detail="Target DXF file not found.")
        
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    allowed_layers = {"STEELWORK", "VERTEX_LABELS"}
    for layer in doc.layers:
        if layer.dxf.name not in allowed_layers:
            layer.off() 

    all_scene_points = []
    for entity in msp.query('*[layer=="STEELWORK"]'):
        entity.dxf.true_color = 0x000000
        if entity.dxf.hasattr('color'):
            entity.dxf.discard('color')
            
        entity.dxf.linetype = "CONTINUOUS"
        entity.dxf.lineweight = 100
            
        all_scene_points.extend(extract_entity_vertices(entity))
        
    if not all_scene_points:
        raise HTTPException(status_code=422, detail="No visible structural framework found to frame image snippet.")
        
    scene_arr = np.array(all_scene_points)
    s_min_x, s_min_y = np.min(scene_arr, axis=0)
    s_max_x, s_max_y = np.max(scene_arr, axis=0)
    
    margin = 500.0  
    width = (s_max_x - s_min_x) + (margin * 2)
    height = (s_max_y - s_min_y) + (margin * 2)
    
    render_box = BoundingBox([
        (s_min_x - margin, s_min_y - margin), 
        (s_max_x + margin, s_max_y + margin)
    ])
    
    max_drawing_units = max(width, height)
    calculated_dpi = int((max_edge_pixels / max_drawing_units) * 72)
    calculated_dpi = max(min(calculated_dpi, 300), 45)
    
    config = Configuration(
        background_policy=BackgroundPolicy.CUSTOM,
        custom_bg_color="#FFFFFF",
        min_lineweight=0.8,        
        lineweight_scaling=2.0     
    )
    
    ctx = RenderContext(doc)
    backend = pymupdf.PyMuPdfBackend()
    
    Frontend(ctx, backend, config=config).draw_layout(msp)
    
    ppm_bytes = backend.get_pixmap_bytes(
        page=layout.Page(0, 0),
        fmt="ppm",
        dpi=calculated_dpi,
        render_box=render_box
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    dxf_filename = os.path.basename(dxf_path)  
    dxf_name = os.path.splitext(dxf_filename)[0]
    output_png_path = f"{output_dir}/{dxf_name}_isolated.png"
    
    with io.BytesIO(ppm_bytes) as stream:
        with Image.open(stream, formats=["ppm"]) as img:
            img.save(output_png_path, format="PNG")
            
    return output_png_path

async def snap_steel_antenna_label(
    dxf_path: str, 
    allowed_points: list = None,
    target_antenna: str = None,
    output_dir: str = "dxf_snap", 
    max_edge_pixels: int = 2048
) -> str:
    """
    Renders a high-contrast PNG snippet displaying ONLY the STEELWORK 
    and the new VERTEX_LABELS layer, isolating it completely from old equipment clutter.
    """
    if not os.path.exists(dxf_path):
        raise HTTPException(status_code=404, detail="Target DXF file not found.")
        
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    allowed_layers = {"STEELWORK", "VERTEX_LABELS", "ANTENNAS DISHES", "ANTENNA_LABELS"}
    for layer in doc.layers:
        if layer.dxf.name not in allowed_layers:
            layer.off()

    valid_vertex_labels = set()
    if allowed_points:
        for pair in allowed_points:
            for point in pair:
                valid_vertex_labels.add(str(point).strip().upper())

    valid_antenna_label = target_antenna.strip().upper() if target_antenna else None

    for entity in list(msp):
        entity_layer = entity.dxf.layer
        entity_type = entity.dxftype()

        if entity_layer == "VERTEX_LABELS":
            if entity_type in ('TEXT', 'MTEXT'):
                entity_text = entity.dxf.text.strip().upper() if entity_type == 'TEXT' else entity.text.strip().upper()
                if valid_vertex_labels and (entity_text not in valid_vertex_labels):
                    msp.delete_entity(entity)
            elif entity_type in ('LWPOLYLINE', 'POLYLINE', 'POINT', 'CIRCLE', 'SOLID', 'INSERT'):
                msp.delete_entity(entity)

        elif entity_layer == "ANTENNA_LABELS":
            if entity_type in ('TEXT', 'MTEXT'):
                entity_text = entity.dxf.text.strip().upper() if entity_type == 'TEXT' else entity.text.strip().upper()
                if valid_antenna_label and (entity_text != valid_antenna_label):
                    msp.delete_entity(entity)
            elif entity_type in ('LWPOLYLINE', 'POLYLINE', 'POINT', 'CIRCLE', 'SOLID', 'INSERT'):
                msp.delete_entity(entity)

    all_scene_points = []
    for entity in msp.query('*[layer=="STEELWORK"]'):
        all_scene_points.extend(extract_entity_vertices(entity))
        
    if not all_scene_points:
        raise HTTPException(status_code=422, detail="No visible structural framework found to frame image snippet.")
        
    scene_arr = np.array(all_scene_points)
    s_min_x, s_min_y = np.min(scene_arr, axis=0)
    s_max_x, s_max_y = np.max(scene_arr, axis=0)
    
    margin = 500.0  
    width = (s_max_x - s_min_x) + (margin * 2)
    height = (s_max_y - s_min_y) + (margin * 2)
    
    render_box = BoundingBox([
        (s_min_x - margin, s_min_y - margin), 
        (s_max_x + margin, s_max_y + margin)
    ])
    
    max_drawing_units = max(width, height)
    calculated_dpi = int((max_edge_pixels / max_drawing_units) * 72)
    calculated_dpi = max(min(calculated_dpi, 300), 45)
    
    ctx = RenderContext(doc)
    backend = pymupdf.PyMuPdfBackend()
    Frontend(ctx, backend).draw_layout(msp)
    
    ppm_bytes = backend.get_pixmap_bytes(
        page=layout.Page(0, 0),
        fmt="ppm",
        dpi=calculated_dpi,
        render_box=render_box
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    dxf_filename = os.path.basename(dxf_path)  
    dxf_name = os.path.splitext(dxf_filename)[0]
    output_png_path = f"{output_dir}/{dxf_name}_point_antenna_isolated.png"
    
    with io.BytesIO(ppm_bytes) as stream:
        with Image.open(stream, formats=["ppm"]) as img:
            img.save(output_png_path, format="PNG")
            
    return output_png_path

async def get_point_pair(image_path: str):
    Prompt = """ 
    You are a mechanical/structural drawing analysis agent. You are given a top-view sketch of a telecom tower steelwork mechanism. On the sketch, several edge points are marked with colored dots and text labels.
    Your job is to output every pair of points that are directly physically connected by a rigid structural member visible in the drawing.
    Return ONLY a JSON array of point pairs. No extra commentary. [ ["P1", "P4"], ["P4", "P5"] ]
    """

    try:    
        with open(image_path, "rb") as f:
            file_bytes = f.read()
            
        filename = os.path.basename(image_path)
        media_type, _ = mimetypes.guess_type(image_path)
        if not media_type:
            media_type = "image/jpeg" 
            
        files_data = [
            (filename, file_bytes, media_type)
        ]

        response_text = await call_claude(Prompt, files_data)

        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
            
        clean_text = clean_text.strip()
        parsed_json = json.loads(clean_text)

        return parsed_json

    except Exception as e:
        return {"status": "failure", "reason": str(e)}

async def get_pair_for_noraml_llm(image_path):
    prompt = """
    You are an expert structural analysis AI specialized in interpreting engineering sketches, blueprints, and top-down CAD diagrams of telecommunications headframe towers.

YOUR PRIMARY TASK:
Identify which structural rail segment a specific labeled antenna is physically attached to and return the exact pair of node/post labels that define that rail segment.

STRUCTURAL ANALYSIS RULES:
1. IDENTIFY THE TARGET ANTENNA:
   - Locate the target antenna label (e.g., "antenna_5").

2. TRACE THE PHYSICAL MOUNT:
   - Locate the primary mounting post (e.g., P6) directly attached to the target antenna via the mounting line/bracket.

3. DETERMINE THE CORRECT RAIL SEGMENT PAIR (CRITICAL STEP):
   - Spatial Proximity vs. Structural Continuity: Do NOT select a pair merely because two nodes are close in Euclidean space (e.g., corner nodes like P3 and P6).
   - Identify the continuous structural beam/rail on which the antenna or its mounting bracket is aligned.
   - Trace the entire beam span from the primary mounting node (e.g., P6) across the frame face to its opposite end-node (e.g., P5) to define the full boundary rail (e.g., ["P6", "P5"]).
   - Distinguish between outer boundary face rails (e.g., P6–P5) and leg/side diagonal rails (e.g., P3–P2 or P6–P3).

4. VERIFICATION CHECK:
   - Verify if the antenna orientation and mounting face face toward/along the top rail, side rail, or bottom rail.
   - Confirm that both nodes in the output pair belong to the SAME continuous structural rail element.

OUTPUT FORMAT:
Respond ONLY with a valid, strict JSON object. Do not include markdown headers, surrounding text, or explanations outside the JSON unless explicitly requested.

{
   "target_antenna": "<antenna_label>",
   "associated_pair": ["<Node1>", "<Node2>"]
}
     """

    try:
        if not os.path.exists(image_path):
             return {"status": "failure", "reason": f"File not found at {image_path}"}

        with open(image_path, "rb") as f:
            file_bytes = f.read() 
            
        filename = os.path.basename(image_path)
        content_type = "image/png"
        formatted_files_data = [(filename, file_bytes, content_type)]

        response_text = await call_claude(prompt, formatted_files_data)

        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
            
        clean_text = clean_text.strip()
        parsed_json = json.loads(clean_text)

        return parsed_json

    except Exception as e:
        return {"status": "failure at normal line for mcp", "reason": str(e)}

# def project_point_to_segment(p: Vec2, a: Vec2, b: Vec2) -> Tuple[Vec2, float]:
#     """
#     Finds the closest point on a line segment AB to point P, 
#     and returns that closest point and the distance.
#     """
#     ab = b - a
#     ap = p - a
    
#     ab_len_sq = ab.dot(ab)
#     if ab_len_sq == 0:
#         return a, p.distance(a) 
        
#     t = ap.dot(ab) / ab_len_sq
#     t = max(0.0, min(1.0, t))
    
#     closest_point = a + ab * t
#     return closest_point, p.distance(closest_point)


def project_point_to_segment(p: Vec2, a: Vec2, b: Vec2) -> Tuple[Vec2, float]:
    """
    Finds the perpendicular (normal) foot of point P onto the *full line*
    defined by A and B, and returns that point and the distance.

    NOTE: intentionally NOT clamped to [0, 1]. Clamping snaps the result to
    whichever endpoint (A or B) is nearest whenever the true perpendicular
    foot falls outside the A-B span - which is what was causing the drop
    to land on an edge/vertex point instead of being normal to the line.
    """
    ab = b - a
    ap = p - a
    
    ab_len_sq = ab.dot(ab)
    if ab_len_sq == 0:
        return a, p.distance(a) 
        
    t = ap.dot(ab) / ab_len_sq
    
    closest_point = a + ab * t
    return closest_point, p.distance(closest_point)



async def drop_normal_to_entity(
    dxf_path: str, 
    radiation_source: Tuple[float, float],
    associated_pair: list[str] = None,  
    vertex_dictionary: dict = None,     
    draw_normal_line: bool = True,
    output_layer: str = "RADIATION_NORMAL"
) -> Dict[str, Any]:
    """
    Calculates the shortest distance from the radiation source to the selected DXF entity 
    OR an associated structural vertex pair (line segment).
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    p_source = Vec2(radiation_source)
    closest_pt = None
    min_dist = float('inf')
    dxftype_evaluated = "UNKNOWN"

    if associated_pair and vertex_dictionary:
        try:
            p1_label, p2_label = associated_pair[0], associated_pair[1]
            coord_a = vertex_dictionary[p1_label]
            coord_b = vertex_dictionary[p2_label]
            
            a = Vec2(coord_a[0], coord_a[1])
            b = Vec2(coord_b[0], coord_b[1])
            
            closest_pt, min_dist = project_point_to_segment(p_source, a, b)
            dxftype_evaluated = "STRUCTURAL_POINT_PAIR_LINE"
        except KeyError as e:
            raise ValueError(f"Vertex label {str(e)} not found in vertex_dictionary.")
    else:
        raise ValueError("Either 'associated_pair' or 'selected_entity' must be provided.")

    if draw_normal_line and closest_pt is not None:
        if output_layer not in doc.layers:
            doc.layers.new(name=output_layer, dxfattribs={'color': 4}) 
            
        msp.add_line(
            start=(p_source.x, p_source.y), 
            end=(closest_pt.x, closest_pt.y),
            dxfattribs={'layer': output_layer}
        )
        doc.save()

    return {
        "shortest_distance": min_dist,
        "intersection_point": [closest_pt.x, closest_pt.y] if closest_pt else None,
        "dxftype_evaluated": dxftype_evaluated
    }


def _copy_entity_with_offset(
    msp,
    entity,
    dx: float,
    dy: float,
    target_layer: str,
    rotation_deg: float = 0.0,
    rotation_center: tuple = (0.0, 0.0),
):
    """Deep-copies a single DXF entity and applies rotation and translation."""
    _ang = math.radians(rotation_deg)
    _cos = math.cos(_ang)
    _sin = math.sin(_ang)
    _rcx, _rcy = rotation_center

    def _rot_pt(px: float, py: float) -> tuple:
        ox, oy = px - _rcx, py - _rcy
        return _rcx + ox * _cos - oy * _sin, _rcy + ox * _sin + oy * _cos

    def _rot_vec(vx: float, vy: float) -> tuple:
        return vx * _cos - vy * _sin, vx * _sin + vy * _cos

    try:
        new_ent = entity.copy()
    except Exception:
        return None

    new_ent.dxf.layer = target_layer
    etype = new_ent.dxftype()

    try:
        if etype == 'LINE':
            s = new_ent.dxf.start
            e = new_ent.dxf.end
            sx, sy = _rot_pt(s.x, s.y)
            ex, ey = _rot_pt(e.x, e.y)
            new_ent.dxf.start = (sx + dx, sy + dy, getattr(s, 'z', 0))
            new_ent.dxf.end   = (ex + dx, ey + dy, getattr(e, 'z', 0))

        elif etype in ('LWPOLYLINE', 'POLYLINE'):
            pts = list(new_ent.get_points())
            new_pts = []
            for p in pts:
                p = tuple(p)
                rx, ry = _rot_pt(p[0], p[1])
                new_pts.append((rx + dx, ry + dy) + p[2:])
            new_ent.set_points(new_pts)

        elif etype == 'CIRCLE':
            c = new_ent.dxf.center
            cx, cy = _rot_pt(c.x, c.y)
            new_ent.dxf.center = (cx + dx, cy + dy, getattr(c, 'z', 0))

        elif etype == 'ARC':
            c = new_ent.dxf.center
            cx, cy = _rot_pt(c.x, c.y)
            new_ent.dxf.center = (cx + dx, cy + dy, getattr(c, 'z', 0))
            new_ent.dxf.start_angle = (new_ent.dxf.start_angle + rotation_deg) % 360.0
            new_ent.dxf.end_angle   = (new_ent.dxf.end_angle   + rotation_deg) % 360.0

        elif etype == 'ELLIPSE':
            c = new_ent.dxf.center
            cx, cy = _rot_pt(c.x, c.y)
            new_ent.dxf.center = (cx + dx, cy + dy, getattr(c, 'z', 0))
            ma = new_ent.dxf.major_axis
            mx, my = _rot_vec(ma.x, ma.y)
            new_ent.dxf.major_axis = (mx, my, getattr(ma, 'z', 0))

        elif etype == 'INSERT':
            ins = new_ent.dxf.insert
            ix, iy = _rot_pt(ins.x, ins.y)
            new_ent.dxf.insert = (ix + dx, iy + dy, getattr(ins, 'z', 0))
            current_rot = getattr(new_ent.dxf, 'rotation', 0.0) or 0.0
            new_ent.dxf.rotation = (current_rot + rotation_deg) % 360.0
        else:
            return None

        msp.add_entity(new_ent)
        return new_ent

    except Exception:
        return None

def _angle_diff_deg(a1: float, a2: float) -> float:
    """Returns the smallest absolute angular difference between two compass azimuths (degrees)."""
    diff = (a1 - a2 + 180.0) % 360.0 - 180.0
    return abs(diff)

async def optimize_radiation_placement_dynamic(
    dxf_path: str,
    radiation_source: List[float],          
    point_pair: List[List[str]],            
    coordinates_map: Dict[str, List[float]],  
    normal_distance: float,
    azimuths: List[float],
    ref_azimuth: float = None,
    steps: int = 20,                         
    matched_index: str = None,              
    clusters_dict: dict = None,
    max_azimuth_deviation: float = 60.0
) -> List[Dict[str, Any]]:
    """
    Finds a feasible placement for the antenna radiation zone by checking ALL provided
    steelwork segment pairs for each azimuth. 
    """
    try:
        doc = ezdxf.readfile(dxf_path)
    except Exception as e:
        raise ValueError(f"Error reading DXF file: {e}")
        
    msp = doc.modelspace()
    rad_arc = None
    rad_fallback_pts = []

    for entity in msp.query('*[layer=="ANTENNA_RADIATION"]'):
        dxftype = entity.dxftype()
        if dxftype == 'ARC':
            rad_arc = entity                  
        elif dxftype in ('LWPOLYLINE', 'POLYLINE'):
            for pt in entity.get_points():
                rad_fallback_pts.append([pt[0], pt[1]])
        elif dxftype == 'LINE':
            rad_fallback_pts.append([entity.dxf.start.x, entity.dxf.start.y])
            rad_fallback_pts.append([entity.dxf.end.x,   entity.dxf.end.y])

    if rad_arc is None and not rad_fallback_pts:
        raise ValueError("Could not find any existing baseline geometry on 'ANTENNA_RADIATION' layer.")

    if rad_arc is not None:
        cx  = rad_arc.dxf.center.x
        cy  = rad_arc.dxf.center.y
        r   = rad_arc.dxf.radius
        start_ang = np.radians(rad_arc.dxf.start_angle)
        end_ang   = np.radians(rad_arc.dxf.end_angle)
        if end_ang <= start_ang:              
            end_ang += 2 * np.pi

        arc_pts = [
            [cx + r * np.cos(a), cy + r * np.sin(a)]
            for a in np.linspace(start_ang, end_ang, num=32)
        ]
        base_poly = Polygon([[cx, cy]] + arc_pts)
        if not base_poly.is_valid:
            base_poly = base_poly.buffer(0)
    else:
        unique_pts = []
        for pt in rad_fallback_pts:
            if pt not in unique_pts:
                unique_pts.append(pt)
        if radiation_source not in unique_pts:
            unique_pts.insert(0, radiation_source)
        base_poly = Polygon(unique_pts)

    base_source_pt = Point(radiation_source[0], radiation_source[1])

    if ref_azimuth is None:
        src_np = np.array([radiation_source[0], radiation_source[1]])
        base_coords_raw = list(base_poly.exterior.coords)[:-1]  
        non_apex = [
            np.array([x, y]) for (x, y) in base_coords_raw
            if np.linalg.norm(np.array([x, y]) - src_np) > 1.0
        ]
        if non_apex:
            centroid_np = np.mean(non_apex, axis=0)
            ref_math_deg = math.degrees(math.atan2(
                centroid_np[1] - src_np[1],
                centroid_np[0] - src_np[0],
            ))
            ref_azimuth = (90.0 - ref_math_deg) % 360.0
        else:
            ref_azimuth = 0.0 

    obstacles: List[Any] = []
    target_layers = {"ANTENNAS_DISHES", "ANTENNAS DISHES", "ANTENNA_DISHES", "ANTENNA DISHES", "FUTURE INSTALLATIONS"}

    steelwork_pts = []
    
    def _extract_dxf_points(entity):
        dxftype = entity.dxftype()
        if dxftype in ('LWPOLYLINE', 'POLYLINE'):
            for pt in entity.get_points():
                steelwork_pts.append((pt[0], pt[1]))
        elif dxftype == 'LINE':
            steelwork_pts.append((entity.dxf.start.x, entity.dxf.start.y))
            steelwork_pts.append((entity.dxf.end.x, entity.dxf.end.y))
        elif dxftype == 'CIRCLE':
            steelwork_pts.append((entity.dxf.center.x, entity.dxf.center.y))
        elif dxftype == 'ARC':
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_ang = np.radians(entity.dxf.start_angle)
            end_ang = np.radians(entity.dxf.end_angle)
            steelwork_pts.append((center.x, center.y))
            steelwork_pts.append((center.x + radius * np.cos(start_ang), center.y + radius * np.sin(start_ang)))
            steelwork_pts.append((center.x + radius * np.cos(end_ang), center.y + radius * np.sin(end_ang)))
        elif dxftype == 'ELLIPSE':
            steelwork_pts.append((entity.dxf.center.x, entity.dxf.center.y))
        elif dxftype in ('SOLID', '3DFACE', 'TRACE'):
            for attr in ('vtx0', 'vtx1', 'vtx2', 'vtx3'):
                v = getattr(entity.dxf, attr, None)
                if v is not None:
                    steelwork_pts.append((v.x, v.y))
        elif dxftype == 'POINT':
            steelwork_pts.append((entity.dxf.location.x, entity.dxf.location.y))
        elif dxftype == 'HATCH':
            for path in entity.paths:
                for vertex in path.vertices:
                    steelwork_pts.append((vertex[0], vertex[1]))
        elif dxftype == 'INSERT':
            try:
                for sub_entity in entity.virtual_entities():
                    _extract_dxf_points(sub_entity)
            except Exception:
                pass

    for entity in msp.query('*[layer=="STEELWORK"]'):
        _extract_dxf_points(entity)

    if steelwork_pts:
        hf_center = np.mean(steelwork_pts, axis=0)
    else:
        all_coords = list(coordinates_map.values())
        hf_center = np.mean(all_coords, axis=0) if all_coords else np.array([0.0, 0.0])

    def _collect_obstacle_geoms(entity, geoms):
        dxftype = entity.dxftype()
        if dxftype in ('LWPOLYLINE', 'POLYLINE'):
            pts = [(p[0], p[1]) for p in entity.get_points()]
            if len(pts) >= 3:
                try:
                    poly = Polygon(pts)
                    geoms.append(poly if poly.is_valid else poly.buffer(0))
                except Exception:
                    geoms.append(LineString(pts).buffer(0.05))
            elif len(pts) == 2:
                geoms.append(LineString(pts).buffer(0.05))
        elif dxftype == 'LINE':
            line = LineString([(entity.dxf.start.x, entity.dxf.start.y), (entity.dxf.end.x, entity.dxf.end.y)])
            geoms.append(line.buffer(0.05))
        elif dxftype == 'CIRCLE':
            center = entity.dxf.center
            radius = entity.dxf.radius
            geoms.append(Point(center.x, center.y).buffer(radius))
        elif dxftype == 'ARC':
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_ang = np.radians(entity.dxf.start_angle)
            end_ang = np.radians(entity.dxf.end_angle)
            if end_ang <= start_ang:
                end_ang += 2 * np.pi
            arc_pts = [
                (center.x + radius * np.cos(a), center.y + radius * np.sin(a))
                for a in np.linspace(start_ang, end_ang, num=16)
            ]
            geoms.append(LineString(arc_pts).buffer(0.05))
        elif dxftype == 'ELLIPSE':
            try:
                pts = [(p.x, p.y) for p in entity.flattening(0.1)]
                if len(pts) >= 2:
                    geoms.append(LineString(pts).buffer(0.05))
            except Exception:
                pass
        elif dxftype in ('SOLID', '3DFACE', 'TRACE'):
            try:
                pts = []
                for attr in ('vtx0', 'vtx1', 'vtx2', 'vtx3'):
                    v = getattr(entity.dxf, attr, None)
                    if v is not None:
                        pts.append((v.x, v.y))
                if len(pts) >= 3:
                    poly = Polygon(pts)
                    geoms.append(poly if poly.is_valid else poly.buffer(0))
            except Exception:
                pass
        elif dxftype == 'POINT':
            loc = entity.dxf.location
            geoms.append(Point(loc.x, loc.y).buffer(0.05))
        elif dxftype == 'INSERT':
            try:
                for sub_entity in entity.virtual_entities():
                    _collect_obstacle_geoms(sub_entity, geoms)
            except Exception:
                pass
        elif dxftype == 'HATCH':
            try:
                for path in entity.paths:
                    pts = [(v[0], v[1]) for v in path.vertices]
                    if len(pts) >= 3:
                        poly = Polygon(pts)
                        geoms.append(poly if poly.is_valid else poly.buffer(0))
            except Exception:
                pass

    for entity in msp:
        if entity.dxf.layer in target_layers:
            _collect_obstacle_geoms(entity, obstacles)

    def _build_extended_cone(source_np: np.ndarray, candidate_poly: Polygon, ext_dist: float) -> Polygon:
        coords = list(candidate_poly.exterior.coords)[:-1]  
        non_apex = [
            np.array([x, y]) for x, y in coords
            if np.linalg.norm(np.array([x, y]) - source_np) > 1e-6
        ]
        if not non_apex:
            return candidate_poly  
        centroid_arr = np.mean(non_apex, axis=0)
        main_angle = np.arctan2(centroid_arr[1] - source_np[1], centroid_arr[0] - source_np[0])

        half_angle = 0.0
        for pt in non_apex:
            vec = pt - source_np
            v_angle = np.arctan2(vec[1], vec[0])
            diff = ((v_angle - main_angle + np.pi) % (2.0 * np.pi)) - np.pi
            if abs(diff) > half_angle:
                half_angle = abs(diff)

        half_angle = half_angle * 1.05 + np.radians(1.0)
        half_angle = min(half_angle, np.pi) 

        num_fan = 64
        fan_pts = [(float(source_np[0]), float(source_np[1]))]
        for k in range(num_fan + 1):
            t = k / num_fan
            ray_angle = main_angle - half_angle + t * (2.0 * half_angle)
            far = source_np + ext_dist * np.array([np.cos(ray_angle), np.sin(ray_angle)])
            fan_pts.append((float(far[0]), float(far[1])))

        try:
            cone = Polygon(fan_pts)
            return cone if cone.is_valid else cone.buffer(0)
        except Exception:
            return candidate_poly  

    all_map_coords = list(coordinates_map.values())
    if all_map_coords:
        xs = [c[0] for c in all_map_coords]
        ys = [c[1] for c in all_map_coords]
        coord_span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    else:
        coord_span = 1000.0

    ext_dist = coord_span * 10.0  
    _ref_src_np = np.array([radiation_source[0], radiation_source[1]])
    ref_extended_cone = _build_extended_cone(
        source_np=_ref_src_np,
        candidate_poly=base_poly,
        ext_dist=ext_dist,
    )

    MIN_PROXIMITY_DISTANCE = 500.0
    results = []
    proposed_extended_cones: List[Any] = []
    proposed_finite_polys: List[Any] = [] 
    proposed_cluster_geoms: List[Any] = []
    proposed_source_positions: List[np.ndarray] = []

    proposed_layer = "PROPOSED_RADIATION"
    if proposed_layer not in doc.layers:
        doc.layers.new(name=proposed_layer, dxfattribs={"color": 3})  

    proposed_antenna_layer = "PROPOSED_EXTRA_ANTENNA"
    if proposed_antenna_layer not in doc.layers:
        doc.layers.new(name=proposed_antenna_layer, dxfattribs={"color": 4})

    for azimuth in azimuths:
        try:
            angle_float = float(azimuth)
        except (TypeError, ValueError):
            if isinstance(azimuth, dict) and "azimuth" in azimuth:
                angle_float = float(azimuth["azimuth"])
            elif isinstance(azimuth, (list, tuple)) and len(azimuth) > 0:
                angle_float = float(azimuth[0])
            else:
                continue

        rotation_deg = ref_azimuth - angle_float
        rotated_poly = rotate(base_poly, angle=rotation_deg, origin=base_source_pt)

        feasible_position = None
        is_feasible = False
        pair_used = None

        for pair in point_pair:
            p1_label, p2_label = pair[0], pair[1]

            if p1_label not in coordinates_map or p2_label not in coordinates_map:
                continue

            p1_coords = coordinates_map[p1_label]
            p2_coords = coordinates_map[p2_label]
            start_pt = np.array([p1_coords[0], p1_coords[1]])
            end_pt   = np.array([p2_coords[0], p2_coords[1]])

            seg_vector = end_pt - start_pt
            seg_length = np.linalg.norm(seg_vector)
            
            if seg_length > 1e-6:
                normal_A = np.array([-seg_vector[1], seg_vector[0]]) / seg_length
                normal_B = np.array([seg_vector[1], -seg_vector[0]]) / seg_length
                
                mid_pt = start_pt + 0.5 * seg_vector
                test_pos_A = mid_pt + normal_A * normal_distance
                test_pos_B = mid_pt + normal_B * normal_distance
                
                dist_A = np.linalg.norm(test_pos_A - hf_center)
                dist_B = np.linalg.norm(test_pos_B - hf_center)
                
                chosen_normal = normal_A if dist_A > dist_B else normal_B
                normal_offset = chosen_normal * normal_distance

                normal_math_deg = math.degrees(math.atan2(chosen_normal[1], chosen_normal[0]))
                normal_azimuth = (90.0 - normal_math_deg) % 360.0
                deviation = _angle_diff_deg(angle_float, normal_azimuth)

                if deviation > max_azimuth_deviation:
                    continue  
            else:
                normal_offset = np.array([0.0, 0.0])

            for i in range(steps + 1):
                t = i / steps
                base_step_pos = start_pt + t * (end_pt - start_pt)
                current_source_pos = base_step_pos + normal_offset
                
                dx = current_source_pos[0] - radiation_source[0]
                dy = current_source_pos[1] - radiation_source[1]
                candidate_poly = translate(rotated_poly, xoff=dx, yoff=dy)

                extended_cone = _build_extended_cone(
                    source_np=current_source_pos,
                    candidate_poly=candidate_poly,
                    ext_dist=ext_dist,
                )

                candidate_cluster_geoms: List[Any] = []
                if matched_index and clusters_dict and matched_index in clusters_dict:
                    _ang_r = math.radians(rotation_deg)
                    _cos_r, _sin_r = math.cos(_ang_r), math.sin(_ang_r)
                    _rcx, _rcy = radiation_source[0], radiation_source[1]
                    for ent_meta in clusters_dict[matched_index].get("entities", []):
                        raw_verts = ent_meta.get("vertices", [])
                        if not raw_verts:
                            continue
                        transformed = []
                        for px, py in raw_verts:
                            ox, oy = px - _rcx, py - _rcy
                            rx = _rcx + ox * _cos_r - oy * _sin_r
                            ry = _rcy + ox * _sin_r + oy * _cos_r
                            transformed.append((rx + dx, ry + dy))
                        try:
                            if len(transformed) >= 3:
                                geom = Polygon(transformed)
                                candidate_cluster_geoms.append(geom if geom.is_valid else geom.buffer(0))
                            elif len(transformed) == 2:
                                candidate_cluster_geoms.append(LineString(transformed).buffer(1.0))
                            elif len(transformed) == 1:
                                candidate_cluster_geoms.append(Point(transformed[0]).buffer(1.0))
                        except Exception:
                            pass
                            
                if not candidate_cluster_geoms:
                    candidate_cluster_geoms = [candidate_poly]

                collision_detected = False
                for obstacle in obstacles:
                    if candidate_poly.intersects(obstacle) or extended_cone.intersects(obstacle):
                        collision_detected = True
                        break

                if not collision_detected:
                    for prev_cluster_geom in proposed_cluster_geoms:
                        if candidate_poly.intersects(prev_cluster_geom) or extended_cone.intersects(prev_cluster_geom):
                            collision_detected = True
                            break

                if not collision_detected:
                    for cg in candidate_cluster_geoms:
                        if ref_extended_cone.intersects(cg):
                            collision_detected = True
                            break

                if not collision_detected:
                    for prev_ext_cone in proposed_extended_cones:
                        if collision_detected:
                            break
                        for cg in candidate_cluster_geoms:
                            if prev_ext_cone.intersects(cg):
                                collision_detected = True
                                break

                if not collision_detected:
                    candidate_pt = Point(float(current_source_pos[0]), float(current_source_pos[1]))
                    for obstacle in obstacles:
                        if candidate_pt.distance(obstacle) < MIN_PROXIMITY_DISTANCE:
                            collision_detected = True
                            break

                if not collision_detected:
                    for prev_pos in proposed_source_positions:
                        if np.linalg.norm(current_source_pos - prev_pos) < MIN_PROXIMITY_DISTANCE:
                            collision_detected = True
                            break

                if not collision_detected:
                    is_feasible = True
                    pair_used = pair
                    coords = list(candidate_poly.exterior.coords)
                    if len(coords) > 1 and coords[0] == coords[-1]:
                        coords = coords[:-1]
                    vertices = [list(p) for p in coords]

                    feasible_position = {
                        "source_x": float(current_source_pos[0]),
                        "source_y": float(current_source_pos[1]),
                        "polygon_vertices": vertices
                    }

                    proposed_extended_cones.append(extended_cone)
                    proposed_finite_polys.append(candidate_poly) 
                    proposed_source_positions.append(current_source_pos.copy())
                    proposed_cluster_geoms.extend(candidate_cluster_geoms)

                    msp.add_lwpolyline(
                        vertices,
                        dxfattribs={
                            "layer": proposed_layer,
                            "color": 3  
                        }
                    ).closed = True

                    if matched_index and clusters_dict and matched_index in clusters_dict:
                        for ent_meta in clusters_dict[matched_index].get("entities", []):
                            handle = ent_meta.get("id")
                            if not handle:
                                continue
                            try:
                                original_entity = doc.entitydb[handle]
                                _copy_entity_with_offset(
                                    msp,
                                    original_entity,
                                    dx, dy,
                                    proposed_antenna_layer,
                                    rotation_deg=rotation_deg,          
                                    rotation_center=tuple(radiation_source)
                                )
                            except KeyError:
                                pass
                            except Exception:
                                pass
                    break

            if is_feasible:
                break 

        results.append({
            "azimuth": angle_float,
            "feasible": is_feasible,
            "point_pair_used": pair_used,   
            "proposed_location": feasible_position
        })

    try:
        doc.save()
    except Exception:
        pass

    return results

async def get_external_point_pairs(image_path):
    try:
        prompt = """ 
        [Role & Objective] You are an expert Telecom Structural Vision Agent specializing in CAD and sketch analysis for telecom site audits. Your task is to analyze top-view diagrams of telecom headframes and extract specific annotated data points (e.g., P1, P2). You must strictly identify points belonging to the Outer Rail Segments and External Antenna Mounts/Booms, while explicitly filtering out all Inner Bracing/Internal Frame points. [Definitions]

* Outer Rails: The main perimeter segments forming the primary geometric shape (usually a triangle) of the headframe.
* External Mounts/Booms: Structural arms or protrusions extending outward from the main outer rails used to mount antennas or empty poles (60mm).
* Inner Bracing: Cross-members or support segments located entirely within the internal area bounded by the outer rails. [Chain of Thought Reasoning Protocol] Before outputting the final list of points, you must process the image using the following step-by-step structural analysis:

1. Global Shape Recognition: Identify the primary geometric perimeter of the headframe structure (e.g., triangular base). Trace the outermost continuous edges.
2. Node Classification - Perimeter: Scan the edges identified in Step 1. List all labeled points that lie directly on this main boundary. These are your "Outer Rail Points".
3. Node Classification - Protrusions: Look outside the primary perimeter. Identify any structural arms extending outward. List the labeled points on these structures. These are your "External Antenna Mounting Points".
4. Node Classification - Interior: Look inside the primary perimeter. Identify the internal support beams. List the points on these beams. These are your "Inner Bracing Points".
5. Validation & Filtering: Combine the lists from Step 2 and Step 3. Cross-reference with Step 4 to ensure absolutely no inner bracing points are included in the final output. [Output Format] Provide a brief summary of your spatial reasoning, followed by a strictly formatted JSON object. { "reasoning_summary": "Brief explanation of the perimeter trace and excluded internal points.", "outer_rail_points": ["P...", "P..."], "external_mounting_points": ["P...", "P..."], "excluded_inner_points": ["P...", "P..."], "final_combined_target_points": ["P...", "P..."] } 

[Output Rules] - **CRITICAL**: 
The output must be a **strict, valid JSON object only**. Do not include any conversational filler, introductory text, or markdown code blocks outside of the JSON itself. 
- Ensure all keys and string values use double quotes. 
[Output Format JSON Schema] 
{ 
"reasoning_summary": "Brief explanation of the perimeter trace and excluded internal points.", 
"outer_rail_points": ["P...", "P..."], 
"external_mounting_points": ["P...", "P..."], 
"excluded_inner_points": ["P...", "P..."], 
"final_combined_target_points": ["P...", "P..."] 
}

[Execution] Analyze the provided image and execute the protocol.
        """

        if not os.path.exists(image_path):
             return {"status": "failure", "reason": f"File not found at {image_path}"}

        with open(image_path, "rb") as f:
            file_bytes = f.read() 
            
        filename = os.path.basename(image_path)
        content_type = "image/png"
        formatted_files_data = [(filename, file_bytes, content_type)]

        response_text = await call_claude(prompt, formatted_files_data)

        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
            
        clean_text = clean_text.strip()
        parsed_json = json.loads(clean_text)

        return parsed_json
    except Exception as e:
        return {"status": "failure", "reason": str(e)}

async def snapshot_target_points_only(
    dxf_path: str, 
    target_points: list[str], 
    output_dir: str = "dxf_snap", 
    max_edge_pixels: int = 2048
) -> str:
    """
    Renders a high-contrast PNG snippet displaying ONLY the STEELWORK 
    and the specific VERTEX_LABELS passed in the target_points list.
    """
    from ezdxf.addons.drawing.config import Configuration, BackgroundPolicy

    if not os.path.exists(dxf_path):
        raise HTTPException(status_code=404, detail="Target DXF file not found.")
        
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    allowed_layers = {"STEELWORK", "VERTEX_LABELS"}
    for layer in doc.layers:
        if layer.dxf.name not in allowed_layers:
            layer.off() 

    valid_points = {str(pt).strip().upper() for pt in target_points}
    valid_colors = set()

    for entity in msp.query('*[layer=="VERTEX_LABELS"]'):
        if entity.dxftype() in ('TEXT', 'MTEXT'):
            text_val = entity.dxf.text.strip().upper() if entity.dxftype() == 'TEXT' else entity.text.strip().upper()
            if text_val in valid_points:
                if entity.dxf.hasattr('true_color'):
                    valid_colors.add(entity.dxf.true_color)

    for entity in list(msp.query('*[layer=="VERTEX_LABELS"]')):
        entity_type = entity.dxftype()
        if entity_type in ('TEXT', 'MTEXT'):
            text_val = entity.dxf.text.strip().upper() if entity_type == 'TEXT' else entity.text.strip().upper()
            if text_val not in valid_points:
                msp.delete_entity(entity)
        else:
            if not entity.dxf.hasattr('true_color') or entity.dxf.true_color not in valid_colors:
                msp.delete_entity(entity)

    all_scene_points = []
    for entity in msp.query('*[layer=="STEELWORK"]'):
        entity.dxf.true_color = 0x000000
        if entity.dxf.hasattr('color'):
            entity.dxf.discard('color')
            
        entity.dxf.linetype = "CONTINUOUS"
        entity.dxf.lineweight = 100
        all_scene_points.extend(extract_entity_vertices(entity))
        
    if not all_scene_points:
        raise HTTPException(status_code=422, detail="No visible structural framework found to frame image snippet.")
        
    scene_arr = np.array(all_scene_points)
    s_min_x, s_min_y = np.min(scene_arr, axis=0)
    s_max_x, s_max_y = np.max(scene_arr, axis=0)
    
    margin = 500.0  
    width = (s_max_x - s_min_x) + (margin * 2)
    height = (s_max_y - s_min_y) + (margin * 2)
    
    render_box = BoundingBox([
        (s_min_x - margin, s_min_y - margin), 
        (s_max_x + margin, s_max_y + margin)
    ])
    
    max_drawing_units = max(width, height)
    calculated_dpi = int((max_edge_pixels / max_drawing_units) * 72)
    calculated_dpi = max(min(calculated_dpi, 300), 45)
    
    config = Configuration(
        background_policy=BackgroundPolicy.CUSTOM,
        custom_bg_color="#FFFFFF",
        min_lineweight=0.8,        
        lineweight_scaling=2.0     
    )
    
    ctx = RenderContext(doc)
    backend = pymupdf.PyMuPdfBackend()
    
    Frontend(ctx, backend, config=config).draw_layout(msp)
    
    ppm_bytes = backend.get_pixmap_bytes(
        page=layout.Page(0, 0),
        fmt="ppm",
        dpi=calculated_dpi,
        render_box=render_box
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    dxf_filename = os.path.basename(dxf_path)  
    dxf_name = os.path.splitext(dxf_filename)[0]
    output_png_path = f"{output_dir}/{dxf_name}_target_points_only.png"
    
    with io.BytesIO(ppm_bytes) as stream:
        with Image.open(stream, formats=["ppm"]) as img:
            img.save(output_png_path, format="PNG")
            
    return output_png_path

async def final_point_pair(image_path: str):
    try:
        prompt = """ 
You are a structural diagram analyst. You will receive a top-view or isometric
sketch of a telecom headframe with labeled junction points (P1, P2 … Pn).
Your job is to identify which pairs of labeled points are directly connected
by a single, uninterrupted rail segment.

{
  "identified_points": ["P1", "P2", ...],
  "direct_connections": [
    ["P1", "P5"],
    ...
  ],
  "notes": "..."
}
        """

        if not os.path.exists(image_path):
             return {"status": "failure", "reason": f"File not found at {image_path}"}

        with open(image_path, "rb") as f:
            file_bytes = f.read() 
            
        filename = os.path.basename(image_path)
        content_type = "image/png"
        formatted_files_data = [(filename, file_bytes, content_type)]

        response_text = await call_claude(prompt, formatted_files_data)

        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
            
        clean_text = clean_text.strip()
        parsed_json = json.loads(clean_text)

        return parsed_json

    except json.JSONDecodeError as e:
        return {
            "status": "failure", 
            "reason": "Failed to parse JSON from Claude", 
            "raw_output": response_text
        }
    except Exception as e:
        return {"status": "failure", "reason": str(e)}

async def auto_analyze_drawing(file: str, img: str, input_params: str, azimuths: list[str] = []):
    """
    Full pipeline:
    1.  Label DXF clusters  →  snapshot  →  LLM identifies matched antenna index
    2.  Draw radiation cone on matched antenna  (returns source point)
    3.  Label STEELWORK vertices (P-labels) for point_pair resolution
    4.  Color-label STEELWORK entities near the radiation source  →  snapshot
    5.  LLM picks which coloured entity is the structural rail (c-label)
    6.  Compute perpendicular offset from source → chosen rail entity
    7.  For each requested azimuth, find the best collision-free point_pair slot
    """

    labeled_file_path = await label_dxf_clusters(file_path=file)
    snapshot_path     = await label_and_snapshot_dxf(dxf_path=labeled_file_path["output_path"])

    ant = await get_antenna(img, snapshot_path["saved_snapshot"], input_params)
    matched_index = ant.get("matched_index")

    if not (ant.get("status") == "Success" and matched_index):
        return {"status": "Fail", "reason": "LLM did not respond or failed matches"}

    azimuth_val = 0.0
    azimuth_match = re.search(r"azimuth(?: of)?\s*:?\s*(\d+)", input_params, re.IGNORECASE)
    if azimuth_match:
        azimuth_val = float(azimuth_match.group(1))
    else:
        deg_match = re.search(r"(\d+)\s*(?:deg|degree|°)", input_params, re.IGNORECASE)
        if deg_match:
            azimuth_val = float(deg_match.group(1))
        else:
            # 3. Final fallback: Look in Claude's reasoning string
            reason_match = re.search(r"azimuth of (\d+)", ant.get("reasoning", ""), re.IGNORECASE)
            if reason_match:
                azimuth_val = float(reason_match.group(1))

    final_dxf_path, radiation_source = await process_antenna_radiation(
        file_path         = labeled_file_path["output_path"],
        clusters_dict     = labeled_file_path["clusters"],
        matched_index_str = matched_index,
        azimuth_deg       = azimuth_val,
    )
    labeled_file_path["output_path"] = final_dxf_path

    vertex_dictionary    = await label_structural_vertices(file_path=final_dxf_path)
    steelwork_label_path = await snapshot_steelwork_and_labels(final_dxf_path)
    external_points = await get_external_point_pairs(steelwork_label_path)

    if external_points.get("status") == "failure":
        return {"status": "Fail", "reason": f"External points LLM failed: {external_points.get('reason')}"}
        
    target_pts = external_points.get("final_combined_target_points", [])

    filtered_snapshot_path = await snapshot_target_points_only(
        dxf_path=final_dxf_path, 
        target_points=target_pts
    )

    point_pair = await get_point_pair(filtered_snapshot_path)  
    antenna_steel_label_path = await snap_steel_antenna_label(final_dxf_path, allowed_points=point_pair, target_antenna=matched_index)
    get_pair_for_normal = await get_pair_for_noraml_llm(antenna_steel_label_path)

    pair_to_evaluate = get_pair_for_normal["associated_pair"]

    print(f"\n--- DEBUG: LLM selected structural rail {pair_to_evaluate} for normal projection ---\n")

    normal_results = await drop_normal_to_entity(
        dxf_path=final_dxf_path,
        radiation_source=radiation_source, 
        associated_pair=pair_to_evaluate,
        vertex_dictionary=vertex_dictionary,
        draw_normal_line=True
    )

    if isinstance(azimuths, str):
        try:
            parsed_azimuths = json.loads(azimuths)
        except json.JSONDecodeError:
            parsed_azimuths = azimuths.replace("[", "").replace("]", "").replace('"', '').split(",")
    else:
        parsed_azimuths = azimuths

    if isinstance(parsed_azimuths, list) and len(parsed_azimuths) == 1 and isinstance(parsed_azimuths[0], str) and "[" in parsed_azimuths[0]:
        try:
            parsed_azimuths = json.loads(parsed_azimuths[0])
        except json.JSONDecodeError:
            pass

    cleaned_azimuths = []
    for a in parsed_azimuths:
        try:
            cleaned_azimuths.append(float(a))
        except (ValueError, TypeError):
            continue

    placement_results = await optimize_radiation_placement_dynamic(
        dxf_path=final_dxf_path,
        radiation_source=radiation_source,       
        point_pair=point_pair,               
        coordinates_map=vertex_dictionary,                 
        normal_distance=normal_results["shortest_distance"],   
        azimuths=cleaned_azimuths,                  
        ref_azimuth=azimuth_val,                    
        steps=25,                                  
        matched_index=matched_index,               
        clusters_dict=labeled_file_path["clusters"]                                   
    )

    return {
        "status":                "success",
        "saved_file":            final_dxf_path,
        "snapshot":              snapshot_path,
        "antenna_identification": ant,
        "radiation":              radiation_source,
        "vertex_map":             vertex_dictionary,
        "steelwork_labelled_path": steelwork_label_path,
        "cluster":               labeled_file_path["clusters"],
        "Normal_distance":       normal_results["shortest_distance"],
        "Final_result":          placement_results
    }
