"""
UV Island packing and SVG export for print layouts.

Arranges flattened UV clusters into a compact layout suitable for printing.
"""

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from scripts.utils.uv_flatten import ClusterUV


@dataclass
class PackedCluster:
    """A cluster with its packed position."""
    cluster_uv: ClusterUV
    offset_x: float = 0.0
    offset_y: float = 0.0
    scale: float = 1.0

    def get_transformed_uv(self, u: float, v: float) -> Tuple[float, float]:
        """Transform UV coordinate to packed space."""
        return (
            u * self.scale + self.offset_x,
            v * self.scale + self.offset_y
        )


@dataclass
class PackedLayout:
    """Complete packed UV layout."""
    clusters: List[PackedCluster] = field(default_factory=list)
    total_width: float = 0.0
    total_height: float = 0.0
    margin: float = 0.02


def pack_uv_islands(
    clusters_uv: List[ClusterUV],
    margin: float = 0.02,
    normalize: bool = True
) -> PackedLayout:
    """
    Pack UV islands using shelf-packing algorithm.

    Args:
        clusters_uv: List of flattened clusters
        margin: Margin between clusters (in normalized units)
        normalize: If True, normalize output to [0, 1] range

    Returns:
        PackedLayout with positioned clusters
    """
    if not clusters_uv:
        return PackedLayout(margin=margin)

    # Sort clusters by height (descending) for better packing
    sorted_clusters = sorted(
        enumerate(clusters_uv),
        key=lambda x: x[1].height,
        reverse=True
    )

    packed = []
    shelves = []  # List of (y_start, height, current_x)

    for orig_idx, cluster in sorted_clusters:
        w = cluster.width + margin
        h = cluster.height + margin

        # Find a shelf that can fit this cluster
        placed = False
        for i, (shelf_y, shelf_h, shelf_x) in enumerate(shelves):
            if h <= shelf_h:  # Fits on this shelf
                # Place cluster
                packed.append(PackedCluster(
                    cluster_uv=cluster,
                    offset_x=shelf_x + margin / 2,
                    offset_y=shelf_y + margin / 2,
                    scale=1.0
                ))
                shelves[i] = (shelf_y, shelf_h, shelf_x + w)
                placed = True
                break

        if not placed:
            # Start new shelf
            if shelves:
                new_y = max(s[0] + s[1] for s in shelves)
            else:
                new_y = 0

            packed.append(PackedCluster(
                cluster_uv=cluster,
                offset_x=margin / 2,
                offset_y=new_y + margin / 2,
                scale=1.0
            ))
            shelves.append((new_y, h, w))

    # Calculate total bounds
    if packed:
        max_x = max(p.offset_x + p.cluster_uv.width for p in packed)
        max_y = max(p.offset_y + p.cluster_uv.height for p in packed)
    else:
        max_x = max_y = 1.0

    # Normalize to [0, 1] if requested
    if normalize and (max_x > 0 and max_y > 0):
        scale = 1.0 / max(max_x, max_y)
        for p in packed:
            p.offset_x *= scale
            p.offset_y *= scale
            p.scale = scale
        max_x *= scale
        max_y *= scale

    return PackedLayout(
        clusters=packed,
        total_width=max_x,
        total_height=max_y,
        margin=margin
    )


def generate_cluster_color(cluster_id: int, total_clusters: int) -> str:
    """Generate a distinct color for a cluster."""
    import colorsys
    hue = cluster_id / max(total_clusters, 1)
    saturation = 0.7 + 0.3 * (cluster_id % 2)
    lightness = 0.5
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def export_print_svg(
    packed: PackedLayout,
    output_path: str,
    page_width: float = 8.5,
    page_height: float = 11.0,
    units: str = "in",
    show_labels: bool = True,
    stroke_width: float = 0.5,
    show_cut_lines: bool = False,
    show_fold_tabs: bool = False,
) -> str:
    """
    Export packed layout as printable SVG for standard Letter paper.

    Args:
        packed: PackedLayout from pack_uv_islands
        output_path: Path to save SVG file
        page_width: Page width in units (default 8.5 for Letter)
        page_height: Page height in units (default 11 for Letter)
        units: Units for dimensions ("in", "mm", "px")
        show_labels: Whether to show cluster labels
        stroke_width: Line width in points
        show_cut_lines: Show dashed cut lines around shapes
        show_fold_tabs: Show fold tabs for paper assembly

    Returns:
        SVG content as string
    """
    # Convert to pixels (96 DPI standard for SVG)
    dpi = 96
    if units == "in":
        px_per_unit = dpi
    elif units == "mm":
        px_per_unit = dpi / 25.4
    else:
        px_per_unit = 1

    width_px = page_width * px_per_unit
    height_px = page_height * px_per_unit

    # Standard printer margins (0.5 inch on all sides for compatibility)
    margin_px = 0.5 * px_per_unit
    content_width = width_px - 2 * margin_px
    content_height = height_px - 2 * margin_px

    # Scale packed layout to fit content area
    scale = min(content_width / packed.total_width,
                content_height / packed.total_height) if packed.total_width > 0 else 1

    svg_parts = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" ',
        f'width="{page_width}{units}" height="{page_height}{units}" ',
        f'viewBox="0 0 {width_px} {height_px}">',
        f'<style>',
        f'  .face {{ fill: none; stroke: #000; stroke-width: 1.5; stroke-linejoin: round; }}',
        f'  .cut-line {{ fill: none; stroke: #666; stroke-width: 0.75; stroke-dasharray: 6,3; }}',
        f'  .fold-tab {{ fill: #e8e8e8; stroke: #000; stroke-width: 0.5; stroke-dasharray: 3,2; }}',
        f'  .fold-line {{ stroke: #999; stroke-width: 0.5; stroke-dasharray: 2,4; }}',
        f'  .label {{ font-family: Arial, Helvetica, sans-serif; font-size: 9px; fill: #000; font-weight: bold; }}',
        f'  .title {{ font-family: Arial, Helvetica, sans-serif; font-size: 12px; fill: #000; font-weight: bold; }}',
        f'</style>',
        f'<rect width="100%" height="100%" fill="white"/>',
    ]

    # Add title
    svg_parts.append(
        f'<text class="title" x="{margin_px}" y="{margin_px - 10}">UV Layout - Letter Paper (8.5" x 11")</text>'
    )

    n_clusters = len(packed.clusters)
    tab_size = 10  # pixels for fold tabs

    for i, pc in enumerate(packed.clusters):
        cluster = pc.cluster_uv
        color = generate_cluster_color(i, n_clusters)

        # Group for this cluster
        svg_parts.append(f'<g id="cluster-{i}" data-cluster="{i}">')

        # Collect all edge segments for cut lines and fold tabs
        edge_segments = []

        # Draw each face
        for face_idx, face in enumerate(cluster.uv_faces):
            points = []
            face_points_px = []
            for local_v in face:
                global_v = cluster.local_to_global[local_v]
                u, v = cluster.uv_coords[global_v]
                # Transform to packed position
                px = margin_px + (pc.offset_x + u * pc.scale) * scale * content_width / packed.total_width
                py = margin_px + (pc.offset_y + v * pc.scale) * scale * content_height / packed.total_height
                # Flip Y for SVG coordinate system
                py = height_px - py
                points.append(f"{px:.2f},{py:.2f}")
                face_points_px.append((px, py))

            # Collect edges for this face
            for j in range(len(face_points_px)):
                p1 = face_points_px[j]
                p2 = face_points_px[(j + 1) % len(face_points_px)]
                edge_segments.append((p1, p2))

            face_id = cluster.face_ids[face_idx] if face_idx < len(cluster.face_ids) else face_idx
            points_str = " ".join(points)
            # Light fill with bold black outline for visibility
            svg_parts.append(
                f'  <polygon class="face" data-face="{face_id}" '
                f'points="{points_str}" fill="{color}" fill-opacity="0.15"/>'
            )

        # Add fold tabs on boundary edges (simplified - add tabs every few edges)
        if show_fold_tabs and edge_segments:
            for idx, (p1, p2) in enumerate(edge_segments[::3]):  # Every 3rd edge
                # Calculate perpendicular offset for tab
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = math.sqrt(dx * dx + dy * dy)
                if length > 0:
                    # Normalize and rotate 90 degrees
                    nx = -dy / length * tab_size
                    ny = dx / length * tab_size
                    # Create trapezoid tab
                    tab_points = [
                        f"{p1[0]:.2f},{p1[1]:.2f}",
                        f"{p2[0]:.2f},{p2[1]:.2f}",
                        f"{p2[0] + nx * 0.7:.2f},{p2[1] + ny * 0.7:.2f}",
                        f"{p1[0] + nx * 0.7:.2f},{p1[1] + ny * 0.7:.2f}",
                    ]
                    svg_parts.append(
                        f'  <polygon class="fold-tab" points="{" ".join(tab_points)}"/>'
                    )

        # Add cut lines around the cluster boundary
        if show_cut_lines:
            # Draw bounding box as cut line
            min_x = margin_px + pc.offset_x * scale * content_width / packed.total_width
            min_y = margin_px + pc.offset_y * scale * content_height / packed.total_height
            max_x = min_x + cluster.width * pc.scale * scale * content_width / packed.total_width
            max_y = min_y + cluster.height * pc.scale * scale * content_height / packed.total_height
            # Flip Y
            min_y_svg = height_px - max_y
            max_y_svg = height_px - min_y
            svg_parts.append(
                f'  <rect class="cut-line" x="{min_x - 5:.2f}" y="{min_y_svg - 5:.2f}" '
                f'width="{max_x - min_x + 10:.2f}" height="{max_y_svg - min_y_svg + 10:.2f}" rx="2"/>'
            )

        # Add cluster label
        if show_labels:
            # Center of cluster
            cx = margin_px + (pc.offset_x + cluster.width / 2) * scale * content_width / packed.total_width
            cy = margin_px + (pc.offset_y + cluster.height / 2) * scale * content_height / packed.total_height
            cy = height_px - cy
            svg_parts.append(
                f'  <text class="label" x="{cx:.2f}" y="{cy:.2f}" '
                f'text-anchor="middle">Cluster {i}</text>'
            )

        svg_parts.append('</g>')

    # Add print instructions at bottom
    svg_parts.append(
        f'<text class="label" x="{margin_px}" y="{height_px - 10}">'
        f'Print at 100% scale. Cut along dashed lines. Fold tabs inward for assembly.</text>'
    )

    svg_parts.append('</svg>')

    svg_content = '\n'.join(svg_parts)

    # Write to file
    with open(output_path, 'w') as f:
        f.write(svg_content)

    return svg_content


def export_print_pdf(
    packed: PackedLayout,
    output_path: str,
    page_width: float = 8.5,
    page_height: float = 11.0,
    units: str = "in",
    show_labels: bool = True,
    stroke_width: float = 0.5,
    show_cut_lines: bool = True,
    show_fold_tabs: bool = True,
) -> bytes:
    """
    Export packed layout as printer-ready PDF for standard Letter paper.

    Args:
        packed: PackedLayout from pack_uv_islands
        output_path: Path to save PDF file
        page_width: Page width in units (default 8.5 for Letter)
        page_height: Page height in units (default 11 for Letter)
        units: Units for dimensions ("in", "mm", "px")
        show_labels: Whether to show cluster labels
        stroke_width: Line width in points
        show_cut_lines: Show dashed cut lines around shapes
        show_fold_tabs: Show fold tabs for paper assembly

    Returns:
        PDF content as bytes
    """
    import tempfile
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF

    # First generate SVG to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".svg", mode="w") as tmp:
        export_print_svg(
            packed,
            tmp.name,
            page_width=page_width,
            page_height=page_height,
            units=units,
            show_labels=show_labels,
            stroke_width=stroke_width,
            show_cut_lines=show_cut_lines,
            show_fold_tabs=show_fold_tabs,
        )
        tmp_svg_path = tmp.name

    try:
        # Convert SVG to RLG drawing
        drawing = svg2rlg(tmp_svg_path)

        # Render to PDF
        renderPDF.drawToFile(drawing, output_path)

        # Read the PDF file back as bytes
        with open(output_path, 'rb') as f:
            pdf_content = f.read()

        return pdf_content
    finally:
        import os
        os.unlink(tmp_svg_path)


def export_multipage_pdf(
    clusters_uv: List,
    pages: dict,  # {page_num: [ClusterPosition]}
    output_path: str,
    page_width: float = 8.5,
    page_height: float = 11.0,
    show_cut_lines: bool = True,
    show_fold_tabs: bool = True,
    seam_edges: dict = None,  # {cluster_id: [seam_edge_info]}
    fold_edges: dict = None,  # {cluster_id: [fold_edge_info]} with fold_type
    print_scale: float = 1.0,  # inches per 3D unit
    cluster_scales: dict = None,  # {cluster_id: scale_factor (3D units per UV unit)}
) -> bytes:
    """
    Export clusters as multi-page PDF with custom positions and proper scaling.

    Args:
        clusters_uv: List of ClusterUV objects
        pages: Dict mapping page number to list of cluster positions
        output_path: Path to save PDF file
        page_width: Page width in inches (default 8.5 for Letter)
        page_height: Page height in inches (default 11 for Letter)
        show_cut_lines: Show dashed cut lines around shapes
        show_fold_tabs: Show fold tabs for paper assembly
        seam_edges: Seam edge info for each cluster
        fold_edges: Fold edge info with fold_type ('valley' or 'mountain')
        print_scale: Inches per 3D model unit (for sizing)
        cluster_scales: Scale factor for each cluster (3D units per UV unit)

    Returns:
        PDF content as bytes
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib.colors import Color, black, gray

    if cluster_scales is None:
        cluster_scales = {}

    # Create PDF
    c = canvas.Canvas(output_path, pagesize=(page_width * inch, page_height * inch))

    margin = 0.5 * inch
    content_width = page_width * inch - 2 * margin
    content_height = page_height * inch - 2 * margin

    # Sort pages
    sorted_pages = sorted(pages.keys())
    if not sorted_pages:
        sorted_pages = [0]

    for page_num in sorted_pages:
        if page_num > 0:
            c.showPage()

        # Draw page header
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, page_height * inch - margin + 15,
                     f"UV Layout - Page {page_num + 1} of {len(sorted_pages)}")

        # Draw margin guides (light gray dashed)
        c.setStrokeColor(gray)
        c.setDash(3, 3)
        c.rect(margin, margin, content_width, content_height)
        c.setDash()

        # Get clusters for this page
        page_positions = pages.get(page_num, [])

        for pos in page_positions:
            cluster_id = pos.cluster_id if hasattr(pos, 'cluster_id') else pos['cluster_id']
            x_norm = pos.x if hasattr(pos, 'x') else pos['x']
            y_norm = pos.y if hasattr(pos, 'y') else pos['y']

            if cluster_id >= len(clusters_uv):
                continue

            cluster = clusters_uv[cluster_id]

            # Calculate position on page
            cluster_x = margin + x_norm * content_width
            cluster_y = margin + (1 - y_norm) * content_height  # Flip Y for PDF coords

            # Get the proper scale factor for this cluster
            # scale_factor = 3D units per UV unit
            # print_scale = inches per 3D unit
            # So: inches_per_UV_unit = scale_factor * print_scale
            cluster_scale_factor = cluster_scales.get(cluster_id, 1.0)
            inches_per_uv_unit = cluster_scale_factor * print_scale

            # Calculate cluster size in inches (then convert to points)
            # UV dimensions * inches_per_UV_unit * 72 points/inch
            cluster_width = cluster.width * inches_per_uv_unit * inch
            cluster_height = cluster.height * inches_per_uv_unit * inch

            # Generate cluster color
            color = generate_cluster_color(cluster_id, len(clusters_uv))
            r = int(color[1:3], 16) / 255
            g = int(color[3:5], 16) / 255
            b = int(color[5:7], 16) / 255

            # Draw cut line around cluster
            if show_cut_lines:
                c.setStrokeColor(gray)
                c.setDash(4, 2)
                c.setLineWidth(0.5)
                c.rect(cluster_x - 5, cluster_y - cluster_height - 5,
                       cluster_width + 10, cluster_height + 10)
                c.setDash()

            # Draw each face
            c.setStrokeColor(black)
            c.setLineWidth(1.5)

            # Get UV bounding box for proper centering
            uv_min_u = cluster.bbox_min[0]
            uv_min_v = cluster.bbox_min[1]

            for face_idx, face in enumerate(cluster.uv_faces):
                points = []
                for local_v in face:
                    global_v = cluster.local_to_global[local_v]
                    u, v = cluster.uv_coords[global_v]
                    # Transform UV to page coords with proper scaling
                    # Offset by bbox min to start at 0, then scale by inches_per_uv_unit
                    px = cluster_x + (u - uv_min_u) * inches_per_uv_unit * inch
                    py = cluster_y - (v - uv_min_v) * inches_per_uv_unit * inch
                    points.append((px, py))

                if points:
                    # Draw filled polygon
                    path = c.beginPath()
                    path.moveTo(points[0][0], points[0][1])
                    for px, py in points[1:]:
                        path.lineTo(px, py)
                    path.close()

                    c.setFillColor(Color(r, g, b, alpha=0.15))
                    c.drawPath(path, fill=1, stroke=1)

            # Draw seam edges with connection labels
            if seam_edges and cluster_id in seam_edges:
                from reportlab.lib.colors import HexColor
                edge_color = HexColor('#c0392b')

                for edge in seam_edges[cluster_id]:
                    # Get UV coordinates
                    uv1 = edge['v1']
                    uv2 = edge['v2']
                    connects_to = edge['connects_to']

                    # Transform to page coordinates with proper scaling
                    ex1 = cluster_x + (uv1[0] - uv_min_u) * inches_per_uv_unit * inch
                    ey1 = cluster_y - (uv1[1] - uv_min_v) * inches_per_uv_unit * inch
                    ex2 = cluster_x + (uv2[0] - uv_min_u) * inches_per_uv_unit * inch
                    ey2 = cluster_y - (uv2[1] - uv_min_v) * inches_per_uv_unit * inch

                    # Draw dashed line for seam edge
                    c.setStrokeColor(edge_color)
                    c.setLineWidth(1.5)
                    c.setDash(3, 2)
                    c.line(ex1, ey1, ex2, ey2)
                    c.setDash()

                    # Calculate midpoint for label
                    mid_x = (ex1 + ex2) / 2
                    mid_y = (ey1 + ey2) / 2

                    # Calculate perpendicular offset
                    dx = ex2 - ex1
                    dy = ey2 - ey1
                    length = math.sqrt(dx * dx + dy * dy)
                    if length > 0:
                        offset_x = (-dy / length) * 8
                        offset_y = (dx / length) * 8
                    else:
                        offset_x = offset_y = 0

                    # Draw label background - wider for full edge pairing
                    label_text = edge.get('label', f"E{edge.get('my_edge', 0)}↔C{connects_to}-E{edge.get('their_edge', 0)}")
                    c.setFillColor(Color(1, 1, 1, alpha=0.9))
                    c.rect(mid_x + offset_x - 22, mid_y + offset_y - 4, 44, 10, fill=1, stroke=0)

                    # Draw label text
                    c.setFillColor(edge_color)
                    c.setFont("Helvetica-Bold", 5)
                    c.drawCentredString(mid_x + offset_x, mid_y + offset_y, label_text)

            # Draw fold edges (interior edges with fold direction)
            if fold_edges and cluster_id in fold_edges:
                valley_color = Color(0.16, 0.50, 0.73)  # #2980b9 blue
                mountain_color = Color(0.75, 0.22, 0.17)  # #c0392b red

                for edge in fold_edges[cluster_id]:
                    uv1 = edge.get('v1', (0, 0))
                    uv2 = edge.get('v2', (0, 0))
                    fold_type = edge.get('fold_type', 'valley')

                    # Transform to page coordinates
                    fx1 = cluster_x + (uv1[0] - uv_min_u) * inches_per_uv_unit * inch
                    fy1 = cluster_y - (uv1[1] - uv_min_v) * inches_per_uv_unit * inch
                    fx2 = cluster_x + (uv2[0] - uv_min_u) * inches_per_uv_unit * inch
                    fy2 = cluster_y - (uv2[1] - uv_min_v) * inches_per_uv_unit * inch

                    if fold_type == 'valley':
                        # Valley fold (fold in): solid blue line
                        c.setStrokeColor(valley_color)
                        c.setLineWidth(1.0)
                        c.setDash()  # solid line
                    else:
                        # Mountain fold (fold out): dashed red line
                        c.setStrokeColor(mountain_color)
                        c.setLineWidth(1.0)
                        c.setDash(4, 2)

                    c.line(fx1, fy1, fx2, fy2)
                    c.setDash()  # reset dash

            # Draw cluster label
            c.setFillColor(black)
            c.setFont("Helvetica-Bold", 9)
            c.drawCentredString(cluster_x + cluster_width / 2,
                               cluster_y - cluster_height - 18,
                               f"Cluster {cluster_id}")

        # Draw footer with legend
        c.setFont("Helvetica", 8)
        c.setFillColor(gray)
        footer_y = 20

        # Legend for fold lines
        legend_x = margin
        c.drawString(legend_x, footer_y, "Print at 100% scale. Legend:")

        # Valley fold indicator (solid blue line)
        legend_x += 130
        c.setStrokeColor(Color(0.16, 0.50, 0.73))
        c.setLineWidth(1.5)
        c.setDash()
        c.line(legend_x, footer_y + 3, legend_x + 20, footer_y + 3)
        c.setFillColor(gray)
        c.drawString(legend_x + 25, footer_y, "Valley (fold in)")

        # Mountain fold indicator (dashed red line)
        legend_x += 95
        c.setStrokeColor(Color(0.75, 0.22, 0.17))
        c.setDash(4, 2)
        c.line(legend_x, footer_y + 3, legend_x + 20, footer_y + 3)
        c.setDash()
        c.setFillColor(gray)
        c.drawString(legend_x + 25, footer_y, "Mountain (fold out)")

        # Cut line indicator
        legend_x += 110
        c.setStrokeColor(Color(0.90, 0.30, 0.24))
        c.setDash(3, 2)
        c.line(legend_x, footer_y + 3, legend_x + 20, footer_y + 3)
        c.setDash()
        c.setFillColor(gray)
        c.drawString(legend_x + 25, footer_y, "Cut")

    c.save()

    # Read PDF back
    with open(output_path, 'rb') as f:
        return f.read()
