"""
poly.py - Real-Time Low-Poly Webcam Effect with Optional Enhancements

Description:
    Captures live video from the webcam and applies a low-polygon (low-poly) effect in real time.
    Optional enhancements can be toggled via variables at the top of the script.

Usage:
    - Run this script with Python 3.
    - Requires OpenCV (cv2) and NumPy.
    - Press ESC to exit the video window.
    - Press 'f' to freeze/unfreeze the current frame (if FREEZE_FRAME_ENABLED is True).

Configuration:
    - Adjust feature switches and parameters below to enable/disable enhancements.

Author: [Your Name]
Date: [YYYY-MM-DD]
"""

import cv2
import numpy as np
import threading
import queue
import time

# === Feature Switches and Parameters ===

# --- Low-poly effect base configuration ---
POLY_SCALE = 0.20
POLY_MAX_CORNERS = 1200
POLY_QUALITY_LEVEL = 0.01
POLY_MIN_DISTANCE = 15

# --- Visual Enhancements ---
DRAW_EDGES = True  # Draw triangle edges after filling
EDGE_COLOR = (20, 20, 20)  # Edge color (B, G, R)
EDGE_THICKNESS = 1

USE_GRID_POINTS = True      # Add grid points (with jitter) to detected corners
GRID_ROWS = 5
GRID_COLS = 5
GRID_JITTER = 2             # Max jitter in pixels for grid points

ADD_BOUNDARY_POINTS = True  # Always add boundary/corner/edge points

USE_CENTROID_COLOR = False  # Use centroid color instead of average color
BLEND_WITH_ORIGINAL = True  # Blend low-poly with original frame
BLEND_ALPHA = 0.9           # Weight for low-poly frame (0.0-1.0)

# --- Interactive Enhancements ---
ENABLE_TRACKBARS = True     # Enable OpenCV trackbars for real-time parameter tuning

FREEZE_FRAME_ENABLED = True # Allow freezing the current frame with 'f' key

# === Utility Functions ===

def get_delaunay_triangles(rect, points):
    """Performs Delaunay triangulation on a set of points within a rectangle."""
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        # Ensure points are float, as required by OpenCV's Subdiv2D
        subdiv.insert((float(p[0]), float(p[1])))
    return subdiv.getTriangleList()

def average_color(frame, pts):
    """Returns the average color inside the polygon defined by pts."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(pts, dtype=np.int32), 255)
    mean = cv2.mean(frame, mask=mask)
    return tuple(map(int, mean[:3]))

def centroid_color(frame, pts):
    """Returns the color at the centroid of the polygon defined by pts."""
    cx = int(round(np.mean([p[0] for p in pts])))
    cy = int(round(np.mean([p[1] for p in pts])))
    h, w = frame.shape[:2]
    cx = np.clip(cx, 0, w - 1)
    cy = np.clip(cy, 0, h - 1)
    return tuple(int(x) for x in frame[cy, cx])

def draw_polygons_highres(orig_frame, triangles, scale, draw_edges=DRAW_EDGES, edge_color=EDGE_COLOR, edge_thickness=EDGE_THICKNESS, use_centroid_color=USE_CENTROID_COLOR):
    """
    Draws polygons at native/original resolution by mapping triangle coordinates
    from the downscaled image up to the original frame size.
    Optionally draws edges and uses centroid color.
    """
    output = np.zeros_like(orig_frame)
    for t in triangles:
        pts_small = [(t[i], t[i+1]) for i in range(0, 6, 2)]
        pts_orig = [(int(round(x * scale)), int(round(y * scale))) for (x, y) in pts_small]
        # Check if triangle is valid (all points within bounds)
        valid_triangle = True
        for pt_x, pt_y in pts_orig:
            if not (0 <= pt_x < orig_frame.shape[1] and 0 <= pt_y < orig_frame.shape[0]):
                valid_triangle = False
                break
        if not valid_triangle:
            continue
        # Choose color sampling method
        if use_centroid_color:
            color = centroid_color(orig_frame, pts_orig)
        else:
            color = average_color(orig_frame, pts_orig)
        cv2.fillConvexPoly(output, np.array(pts_orig, dtype=np.int32), color)
        if draw_edges:
            cv2.polylines(output, [np.array(pts_orig, dtype=np.int32)], isClosed=True, color=edge_color, thickness=edge_thickness)
    return output

def add_grid_points(w, h, rows, cols, jitter=0):
    """Generates a grid of points with optional jitter."""
    points = []
    for r in range(rows):
        for c in range(cols):
            x = int(c * (w - 1) / (cols - 1))
            y = int(r * (h - 1) / (rows - 1))
            if jitter > 0:
                x = np.clip(x + np.random.randint(-jitter, jitter+1), 0, w-1)
                y = np.clip(y + np.random.randint(-jitter, jitter+1), 0, h-1)
            points.append((x, y))
    return points

def add_boundary_points(w, h):
    """Returns a list of points at the corners and midpoints of the image boundary."""
    return [
        (0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
        (w // 2, 0), (w // 2, h - 1),
        (0, h // 2), (w - 1, h // 2)
    ]

# === Main Low-Poly Effect ===

def low_poly_effect(frame):
    """
    Applies the low-poly effect at a lower resolution for speed,
    but draws polygons at the original/native resolution for sharp output.
    Optional enhancements are controlled by switches at the top.
    """
    # Downscale the frame for faster processing and larger polygons
    small = cv2.resize(frame, (0, 0), fx=POLY_SCALE, fy=POLY_SCALE)
    h_small, w_small = small.shape[:2]
    h_orig, w_orig = frame.shape[:2]
    scale = w_orig / w_small  # Assume uniform scaling

    # Detect corners on the small frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=POLY_MAX_CORNERS,
        qualityLevel=POLY_QUALITY_LEVEL,
        minDistance=POLY_MIN_DISTANCE
    )
    points = []
    if corners is not None:
        points = [(int(p[0][0]), int(p[0][1])) for p in corners]

    # Optionally add grid points (with jitter)
    if USE_GRID_POINTS:
        points.extend(add_grid_points(w_small, h_small, GRID_ROWS, GRID_COLS, GRID_JITTER))

    # Optionally add boundary points
    if ADD_BOUNDARY_POINTS:
        points.extend(add_boundary_points(w_small, h_small))

    # Remove duplicate points
    points = list(set(points))

    if not points:
        return frame.copy()

    rect = (0, 0, w_small, h_small)
    # Filter points to be within the rectangle for Subdiv2D
    points_in_rect = [p for p in points if 0 <= p[0] < w_small and 0 <= p[1] < h_small]
    if len(points_in_rect) < 3:
        return frame.copy()

    tris = get_delaunay_triangles(rect, points_in_rect)

    # Draw polygons (with optional edge drawing and color sampling)
    poly_highres = draw_polygons_highres(
        frame, tris, scale,
        draw_edges=DRAW_EDGES,
        edge_color=EDGE_COLOR,
        edge_thickness=EDGE_THICKNESS,
        use_centroid_color=USE_CENTROID_COLOR
    )

    # Optionally blend with original frame
    if BLEND_WITH_ORIGINAL:
        blended = cv2.addWeighted(poly_highres, BLEND_ALPHA, frame, 1 - BLEND_ALPHA, 0)
        return blended
    else:
        return poly_highres

# === Interactive Controls (Trackbars) ===

def update_trackbar_vars():
    """Updates global variables from trackbars (for real-time tuning)."""
    global POLY_SCALE, POLY_MAX_CORNERS, POLY_QUALITY_LEVEL, POLY_MIN_DISTANCE
    scale = cv2.getTrackbarPos('Scale', 'Low-Poly Webcam')
    POLY_SCALE = max(scale / 100.0, 0.01)
    POLY_MAX_CORNERS = max(cv2.getTrackbarPos('Max Corners', 'Low-Poly Webcam'), 10)
    POLY_QUALITY_LEVEL = max(cv2.getTrackbarPos('Quality', 'Low-Poly Webcam') / 1000.0, 0.001)
    POLY_MIN_DISTANCE = max(cv2.getTrackbarPos('Min Dist', 'Low-Poly Webcam'), 1)

def setup_trackbars():
    """Creates OpenCV trackbars for real-time parameter tuning."""
    cv2.namedWindow('Low-Poly Webcam')
    cv2.createTrackbar('Scale', 'Low-Poly Webcam', int(POLY_SCALE * 100), 100, lambda x: update_trackbar_vars())
    cv2.createTrackbar('Max Corners', 'Low-Poly Webcam', POLY_MAX_CORNERS, 3000, lambda x: update_trackbar_vars())
    cv2.createTrackbar('Quality', 'Low-Poly Webcam', int(POLY_QUALITY_LEVEL * 1000), 100, lambda x: update_trackbar_vars())
    cv2.createTrackbar('Min Dist', 'Low-Poly Webcam', POLY_MIN_DISTANCE, 100, lambda x: update_trackbar_vars())

# === Main Loop ===

cv2.namedWindow('Low-Poly Webcam', cv2.WINDOW_NORMAL)  # Ensure window is created before imshow and trackbars

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")

if ENABLE_TRACKBARS:
    setup_trackbars()

frozen_frame = None
freeze_mode = False
start_time = cv2.getTickCount()
frame_count = 0
fps = 0

# Processing queue and flags
frame_queue = queue.Queue(maxsize=2)  # Allow 2 frames to be queued
result_queue = queue.Queue(maxsize=2)  # Allow 2 results to be queued
processing_active = True
last_ui_check_time = time.time()
last_displayed_frame = None  # Keep track of last displayed processed frame

# Thread for processing frames
def process_frames():
    while processing_active:
        try:
            # Get the latest frame with a timeout
            frame_data = frame_queue.get(timeout=0.1)
            if frame_data is None:
                continue
                
            # Apply effect and put result in queue
            frame, params = frame_data
            poly_frame = low_poly_effect(frame)
            
            # Put in queue, only replacing if full
            try:
                result_queue.put(poly_frame, block=False)
            except queue.Full:
                # Queue is full, get an item and then put
                try:
                    result_queue.get_nowait()
                    result_queue.put(poly_frame, block=False)
                except Exception:
                    pass
        except queue.Empty:
            # No new frame to process, just continue
            continue
        except Exception as e:
            print(f"Processing error: {e}")

# Start processing thread
processing_thread = threading.Thread(target=process_frames)
processing_thread.daemon = True
processing_thread.start()

# --- Main Event Loop with improved display consistency ---
try:
    last_frame = None
    last_params = None
    
    while True:
        # Guarantee UI responsiveness by checking frequently
        key = cv2.waitKey(1) & 0xFF
        
        # Check for key presses immediately
        if key == 27:  # ESC key
            print("ESC pressed, exiting.")
            break
        if FREEZE_FRAME_ENABLED and key == ord('f'):
            freeze_mode = not freeze_mode
            print(f"Freeze mode: {freeze_mode}")
            if freeze_mode:
                frozen_frame = last_frame.copy() if last_frame is not None else None
            else:
                frozen_frame = None
        
        # Force UI event processing every 100ms
        current_time = time.time()
        if current_time - last_ui_check_time > 0.1:
            cv2.waitKey(1)  # Additional UI refresh
            last_ui_check_time = current_time
            
            # Update trackbar variables
            if ENABLE_TRACKBARS:
                update_trackbar_vars()
        
        # Read frame or use frozen frame
        if FREEZE_FRAME_ENABLED and freeze_mode and frozen_frame is not None:
            frame = frozen_frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break
            if FREEZE_FRAME_ENABLED and freeze_mode:
                frozen_frame = frame.copy()
        
        if last_frame is None:
            last_frame = frame.copy()
        
        # Check if parameters changed
        current_params = (POLY_SCALE, POLY_MAX_CORNERS, POLY_QUALITY_LEVEL, POLY_MIN_DISTANCE, 
                         DRAW_EDGES, USE_GRID_POINTS, ADD_BOUNDARY_POINTS, USE_CENTROID_COLOR)
        
        # Queue frame for processing if parameters changed or queue not full
        params_changed = last_params != current_params
        if params_changed or frame_queue.qsize() < 2:
            try:
                # Queue new frame with current parameters without blocking
                frame_queue.put((frame.copy(), current_params), block=False)
                last_params = current_params
            except queue.Full:
                pass  # Queue is full, skip this frame
        
        # Try to get processed frame
        poly_frame = None
        try:
            # Non-blocking get to avoid waiting for processing
            poly_frame = result_queue.get_nowait()
            last_displayed_frame = poly_frame  # Save this successful frame
        except queue.Empty:
            # If no new processed frame, use the last one if available
            if last_displayed_frame is not None:
                poly_frame = last_displayed_frame
            else:
                # First run or processing hasn't completed yet
                # Show original with "Processing..." overlay
                temp_frame = frame.copy()
                cv2.putText(temp_frame, "Processing...", (temp_frame.shape[1]//2-100, temp_frame.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Low-Poly Webcam', temp_frame)
                last_frame = frame.copy()
                continue
        
        # If we have a processed frame to display
        if poly_frame is not None:
            # Calculate and display FPS
            frame_count += 1
            if frame_count >= 10:
                end_time = cv2.getTickCount()
                elapsed_time = (end_time - start_time) / cv2.getTickFrequency()
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = cv2.getTickCount()
            
            # Display FPS and UI responsiveness indicator
            cv2.putText(poly_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(poly_frame, f"Freeze: {freeze_mode}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Low-Poly Webcam', poly_frame)
        
        last_frame = frame.copy()

finally:
    # Clean up resources
    processing_active = False
    if processing_thread.is_alive():
        processing_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
