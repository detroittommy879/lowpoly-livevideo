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
    - Press 'g' to toggle GPU acceleration (if available).

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

# === GPU Acceleration Settings ===
USE_GPU = True  # Whether to use GPU acceleration (if available)
GPU_AVAILABLE = False  # Will be set based on detection
GPU_DETECTION_DONE = False  # Flag to avoid repeated detection
USE_CUDA = False  # Will be set if CUDA is available
USE_OPENCL = False  # Will be set if OpenCL is available

# Check for GPU support
def check_gpu_support():
    global GPU_AVAILABLE, USE_CUDA, USE_OPENCL, GPU_DETECTION_DONE
    
    if GPU_DETECTION_DONE:
        return GPU_AVAILABLE
    
    # Check for CUDA support
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        USE_CUDA = True
        GPU_AVAILABLE = True
        print(f"CUDA is available with {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
    
    # Check for OpenCL support
    try:
        # Check if OpenCL is available and there are devices
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.useOpenCL():
                USE_OPENCL = True
                GPU_AVAILABLE = True
                print("OpenCL is available")
            else:
                print("OpenCL is available but not usable")
        else:
            print("OpenCL is not available")
    except AttributeError:
        print("OpenCL support not included in this OpenCV build")
    
    GPU_DETECTION_DONE = True
    return GPU_AVAILABLE

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
GRID_ROWS = 35
GRID_COLS = 35
GRID_JITTER = 5             # Max jitter in pixels for grid points (must be ≤ 20)

ADD_BOUNDARY_POINTS = True  # Always add boundary/corner/edge points

USE_CENTROID_COLOR = False  # Use centroid color instead of average color
BLEND_WITH_ORIGINAL = True  # Blend low-poly with original frame
BLEND_ALPHA = 0.97           # Weight for low-poly frame (0.0-1.0)

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

def average_color(frame, pts, use_gpu=False):
    """Returns the average color inside the polygon defined by pts."""
    if use_gpu and USE_CUDA:
        # CUDA implementation for mask creation and mean calculation
        # Note: This is a simplified approach, as CUDA requires more setup
        # For complex operations it's often faster to batch process
        # For now we'll keep CPU implementation even with GPU flag for this function
        pass
    
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

def draw_polygons_highres(orig_frame, triangles, scale, draw_edges=DRAW_EDGES, edge_color=EDGE_COLOR, edge_thickness=EDGE_THICKNESS, use_centroid_color=USE_CENTROID_COLOR, use_gpu=False):
    """
    Draws polygons at native/original resolution by mapping triangle coordinates
    from the downscaled image up to the original frame size.
    Optionally draws edges and uses centroid color.
    """
    # Create output on GPU if available
    if use_gpu and USE_CUDA:
        # Initialize output frame
        output = np.zeros_like(orig_frame)
        gpu_output = cv2.cuda_GpuMat(output.shape[0], output.shape[1], cv2.CV_8UC3)
        gpu_output.upload(output)
        
        # Process triangles in batches for GPU efficiency
        batch_size = 100  # Process 100 triangles at a time
        for i in range(0, len(triangles), batch_size):
            batch_triangles = triangles[i:i+batch_size]
            # Process this batch on CPU for now (full GPU implementation would be complex)
            batch_output = np.zeros_like(orig_frame)
            
            for t in batch_triangles:
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
                cv2.fillConvexPoly(batch_output, np.array(pts_orig, dtype=np.int32), color)
                if draw_edges:
                    cv2.polylines(batch_output, [np.array(pts_orig, dtype=np.int32)], isClosed=True, color=edge_color, thickness=edge_thickness)
            
            # Upload batch result to GPU and accumulate
            gpu_batch = cv2.cuda_GpuMat()
            gpu_batch.upload(batch_output)
            
            # Add batch result to output using GPU (max operation combines the polygons)
            cv2.cuda.max(gpu_output, gpu_batch, gpu_output)
        
        # Download final result
        output = gpu_output.download()
        return output
    
    # CPU implementation (original)
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
    # Calculate step sizes to avoid floating point operations in the inner loop
    x_step = (w - 1) / (cols - 1)
    y_step = (h - 1) / (rows - 1)
    
    for r in range(rows):
        for c in range(cols):
            x = int(c * x_step)
            y = int(r * y_step)
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

def low_poly_effect(frame, use_gpu=False):
    """
    Applies the low-poly effect at a lower resolution for speed,
    but draws polygons at the original/native resolution for sharp output.
    Optional enhancements are controlled by switches at the top.
    """
    # Track time for each part of the algorithm
    timing = {}
    timing['start'] = time.time()
    
    # Check if we should use GPU
    if use_gpu:
        gpu_available = check_gpu_support()
        if not gpu_available:
            use_gpu = False
    
    # Downscale the frame for faster processing and larger polygons
    if use_gpu and USE_CUDA:
        # Use CUDA for resizing
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        gpu_small = cv2.cuda.resize(gpu_frame, (0, 0), fx=POLY_SCALE, fy=POLY_SCALE)
        small = gpu_small.download()
    else:
        # Use CPU for resizing
        small = cv2.resize(frame, (0, 0), fx=POLY_SCALE, fy=POLY_SCALE)
    
    h_small, w_small = small.shape[:2]
    h_orig, w_orig = frame.shape[:2]
    scale = w_orig / w_small  # Assume uniform scaling
    
    timing['resize'] = time.time()

    # Detect corners on the small frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    if use_gpu and USE_CUDA:
        # Use GPU for corner detection
        gpu_gray = cv2.cuda_GpuMat()
        gpu_gray.upload(gray)
        
        # Use CUDA goodFeaturesToTrack
        detector = cv2.cuda.createGoodFeaturesToTrackDetector(
            cv2.CV_8UC1,
            POLY_MAX_CORNERS,
            POLY_QUALITY_LEVEL,
            POLY_MIN_DISTANCE
        )
        
        gpu_corners = detector.detect(gpu_gray)
        corners = gpu_corners.download()
    else:
        # Use CPU for corner detection
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=POLY_MAX_CORNERS,
            qualityLevel=POLY_QUALITY_LEVEL,
            minDistance=POLY_MIN_DISTANCE
        )
    
    points = []
    if corners is not None:
        points = [(int(p[0][0]), int(p[0][1])) for p in corners]
    
    timing['corners'] = time.time()

    # Calculate how many points we're adding
    total_points = len(points)
    
    # Optionally add grid points (with jitter)
    if USE_GRID_POINTS:
        # Use a more efficient grid by limiting points based on image resolution
        # Scale grid based on resolution - bigger image should have more points
        effective_rows = min(GRID_ROWS, max(10, int(h_small * 0.15)))
        effective_cols = min(GRID_COLS, max(10, int(w_small * 0.15)))
        
        grid_points = add_grid_points(w_small, h_small, effective_rows, effective_cols, GRID_JITTER)
        points.extend(grid_points)
        total_points += len(grid_points)
    
    # Optionally add boundary points
    if ADD_BOUNDARY_POINTS:
        boundary_points = add_boundary_points(w_small, h_small)
        points.extend(boundary_points)
        total_points += len(boundary_points)
    
    timing['add_points'] = time.time()
    
    # Remove duplicate points
    points = list(set(points))
    
    # Limit total number of points for performance
    max_points = 2000  # Empirically determined good value for real-time performance
    
    if len(points) > max_points:
        # Randomly select limited number of points
        points = [points[i] for i in np.random.choice(len(points), max_points, replace=False)]
    
    timing['filter_points'] = time.time()

    if not points:
        return frame.copy()

    rect = (0, 0, w_small, h_small)
    # Filter points to be within the rectangle for Subdiv2D
    points_in_rect = [p for p in points if 0 <= p[0] < w_small and 0 <= p[1] < h_small]
    if len(points_in_rect) < 3:
        return frame.copy()
    
    timing['before_delaunay'] = time.time()
    
    # Perform the Delaunay triangulation (this is often the bottleneck)
    # No direct GPU implementation for Delaunay in OpenCV
    tris = get_delaunay_triangles(rect, points_in_rect)
    
    timing['after_delaunay'] = time.time()

    # Draw polygons (with optional edge drawing and color sampling)
    poly_highres = draw_polygons_highres(
        frame, tris, scale,
        draw_edges=DRAW_EDGES,
        edge_color=EDGE_COLOR,
        edge_thickness=EDGE_THICKNESS,
        use_centroid_color=USE_CENTROID_COLOR,
        use_gpu=use_gpu
    )
    
    timing['after_drawing'] = time.time()

    # Optionally blend with original frame
    if BLEND_WITH_ORIGINAL:
        if use_gpu and USE_CUDA:
            # Use CUDA for blending
            gpu_poly = cv2.cuda_GpuMat()
            gpu_frame = cv2.cuda_GpuMat()
            
            gpu_poly.upload(poly_highres)
            gpu_frame.upload(frame)
            
            # Use addWeighted to blend the images
            gpu_result = cv2.cuda.addWeighted(gpu_poly, BLEND_ALPHA, gpu_frame, 1 - BLEND_ALPHA, 0)
            result = gpu_result.download()
        else:
            # Use CPU for blending
            blended = cv2.addWeighted(poly_highres, BLEND_ALPHA, frame, 1 - BLEND_ALPHA, 0)
            result = blended
    else:
        result = poly_highres
    
    timing['end'] = time.time()
    
    # Calculate timing breakdowns for debugging
    time_breakdown = {
        'resize': timing['resize'] - timing['start'],
        'corners': timing['corners'] - timing['resize'],
        'add_points': timing['add_points'] - timing['corners'],
        'filter_points': timing['filter_points'] - timing['add_points'],
        'delaunay': timing['after_delaunay'] - timing['before_delaunay'], 
        'drawing': timing['after_drawing'] - timing['after_delaunay'],
        'blend': timing['end'] - timing['after_drawing'],
        'total': timing['end'] - timing['start'],
        'gpu_used': use_gpu and (USE_CUDA or USE_OPENCL)
    }

    # Return result, timing, and point count as a tuple
    return result, time_breakdown, len(points_in_rect)

# === Interactive Controls (Trackbars) ===

def update_trackbar_vars():
    """Updates global variables from trackbars (for real-time tuning)."""
    global POLY_SCALE, POLY_MAX_CORNERS, POLY_QUALITY_LEVEL, POLY_MIN_DISTANCE
    global GRID_ROWS, GRID_COLS, GRID_JITTER, USE_GRID_POINTS
    
    scale = cv2.getTrackbarPos('Scale', 'Low-Poly Webcam')
    POLY_SCALE = max(scale / 100.0, 0.01)
    POLY_MAX_CORNERS = max(cv2.getTrackbarPos('Max Corners', 'Low-Poly Webcam'), 10)
    POLY_QUALITY_LEVEL = max(cv2.getTrackbarPos('Quality', 'Low-Poly Webcam') / 1000.0, 0.001)
    POLY_MIN_DISTANCE = max(cv2.getTrackbarPos('Min Dist', 'Low-Poly Webcam'), 1)
    
    # Grid points controls
    USE_GRID_POINTS = cv2.getTrackbarPos('Use Grid', 'Low-Poly Webcam') == 1
    GRID_ROWS = max(cv2.getTrackbarPos('Grid Rows', 'Low-Poly Webcam'), 5)
    GRID_COLS = max(cv2.getTrackbarPos('Grid Cols', 'Low-Poly Webcam'), 5)
    GRID_JITTER = cv2.getTrackbarPos('Grid Jitter', 'Low-Poly Webcam')

def setup_trackbars():
    """Creates OpenCV trackbars for real-time parameter tuning."""
    # Window is already created in main()
    cv2.createTrackbar('Scale', 'Low-Poly Webcam', int(POLY_SCALE * 100), 100, lambda x: update_trackbar_vars())
    cv2.createTrackbar('Max Corners', 'Low-Poly Webcam', int(POLY_MAX_CORNERS), 3000, lambda x: update_trackbar_vars())
    cv2.createTrackbar('Quality', 'Low-Poly Webcam', int(POLY_QUALITY_LEVEL * 1000), 100, lambda x: update_trackbar_vars())
    cv2.createTrackbar('Min Dist', 'Low-Poly Webcam', int(POLY_MIN_DISTANCE), 100, lambda x: update_trackbar_vars())
    
    # Grid point controls - ensure initial values are within range
    cv2.createTrackbar('Use Grid', 'Low-Poly Webcam', 1 if USE_GRID_POINTS else 0, 1, lambda x: update_trackbar_vars())
    
    # Ensure initial value doesn't exceed maximum for all trackbars
    grid_rows = max(1, min(GRID_ROWS, 50))
    grid_cols = max(1, min(GRID_COLS, 50))
    grid_jitter = max(1, min(GRID_JITTER, 20))
    
    cv2.createTrackbar('Grid Rows', 'Low-Poly Webcam', grid_rows, 50, lambda x: update_trackbar_vars())
    cv2.createTrackbar('Grid Cols', 'Low-Poly Webcam', grid_cols, 50, lambda x: update_trackbar_vars())
    cv2.createTrackbar('Grid Jitter', 'Low-Poly Webcam', grid_jitter, 20, lambda x: update_trackbar_vars())

# === Main Loop ===

# Make sure to check for errors during initialization
def main():
    global USE_GPU  # Ensure all assignments and reads refer to the global variable
    try:
        check_gpu_support()  # Check for GPU availability at startup

        cv2.namedWindow('Low-Poly Webcam', cv2.WINDOW_NORMAL)  # Ensure window is created before imshow and trackbars

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            exit(1)

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

        # Track processing time separately from display time
        process_start_time = time.time()
        process_count = 0
        processing_fps = 0
        display_fps = 0

        # Thread for processing frames
        def process_frames():
            nonlocal process_count, processing_fps, process_start_time, processing_active
            global USE_GPU  # Ensure access to the global USE_GPU variable
            while processing_active:
                try:
                    # Get the latest frame with a timeout
                    frame_data = frame_queue.get(timeout=0.1)
                    if frame_data is None:
                        continue

                    # Measure processing time
                    process_frame_start = time.time()

                    # Apply effect and put result in queue
                    frame, params = frame_data
                    poly_frame, timing_data, point_count = low_poly_effect(frame, use_gpu=USE_GPU and GPU_AVAILABLE)

                    # Calculate processing time and update FPS counter
                    process_time = time.time() - process_frame_start
                    process_count += 1

                    # Update processing FPS every 5 frames
                    if process_count >= 5:
                        current_time = time.time()
                        elapsed = current_time - process_start_time
                        if elapsed > 0:
                            processing_fps = process_count / elapsed
                        process_count = 0
                        process_start_time = current_time

                    # Put result in queue, with performance stats
                    result_data = {
                        'frame': poly_frame,
                        'processing_fps': processing_fps,
                        'process_time': process_time,
                        'timing_data': timing_data,
                        'point_count': point_count
                    }

                    # Put in queue, only replacing if full
                    try:
                        result_queue.put(result_data, block=False)
                    except queue.Full:
                        # Queue is full, get an item and then put
                        try:
                            result_queue.get_nowait()
                            result_queue.put(result_data, block=False)
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
            display_start_time = time.time()
            display_count = 0

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
                if key == ord('g') and GPU_AVAILABLE:
                    # Toggle GPU usage
                    USE_GPU = not USE_GPU
                    print(f"GPU acceleration: {'ON' if USE_GPU else 'OFF'}")

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
                current_params = (
                    POLY_SCALE,
                    POLY_MAX_CORNERS,
                    POLY_QUALITY_LEVEL,
                    POLY_MIN_DISTANCE,
                    DRAW_EDGES,
                    EDGE_COLOR,
                    EDGE_THICKNESS,
                    USE_GRID_POINTS,
                    GRID_ROWS,
                    GRID_COLS,
                    GRID_JITTER,
                    ADD_BOUNDARY_POINTS,
                    USE_CENTROID_COLOR,
                    BLEND_WITH_ORIGINAL,
                    BLEND_ALPHA
                )

                # Collect info about the points for display
                points_info = []
                points_info.append(f"Corners: ~{POLY_MAX_CORNERS}")
                if USE_GRID_POINTS:
                    points_info.append(f"Grid: {GRID_ROWS}x{GRID_COLS}")
                if ADD_BOUNDARY_POINTS:
                    points_info.append("Boundary: 8")

                # Queue frame for processing if parameters changed or frame queue not full
                params_changed = last_params != current_params
                if params_changed or frame_queue.qsize() < 2:
                    try:
                        # Queue new frame with current parameters without blocking
                        frame_queue.put((frame.copy(), current_params), block=False)
                        last_params = current_params
                    except queue.Full:
                        pass  # Queue is full, skip this frame

                # Try to get processed frame
                result_data = None
                try:
                    # Non-blocking get to avoid waiting for processing
                    result_data = result_queue.get_nowait()
                    poly_frame = result_data['frame']
                    processing_fps = result_data['processing_fps']
                    process_time = result_data['process_time']
                    timing_data = result_data.get('timing_data', {})
                    point_count = result_data.get('point_count', 0)
                    last_displayed_frame = poly_frame  # Save this successful frame
                except queue.Empty:
                    # If no new processed frame, use the last one if available
                    if last_displayed_frame is not None:
                        poly_frame = last_displayed_frame
                    else:
                        # First run or processing hasn't completed yet
                        # Show original with "Processing..." overlay
                        temp_frame = frame.copy()
                        cv2.putText(
                            temp_frame,
                            "Processing...",
                            (temp_frame.shape[1] // 2 - 100, temp_frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )
                        cv2.imshow('Low-Poly Webcam', temp_frame)
                        last_frame = frame.copy()
                        continue

                # If we have a processed frame to display
                if poly_frame is not None:
                    # Calculate display FPS (how fast we're showing frames)
                    display_count += 1
                    if display_count >= 10:
                        current_time = time.time()
                        elapsed = current_time - display_start_time
                        if elapsed > 0:
                            display_fps = display_count / elapsed
                        display_count = 0
                        display_start_time = current_time

                    # Display both processing and display FPS
                    cv2.putText(
                        poly_frame,
                        f"Display FPS: {display_fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        poly_frame,
                        f"Processing FPS: {processing_fps:.1f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    # Display GPU status
                    gpu_status = "GPU: ON" if (USE_GPU and GPU_AVAILABLE) else "GPU: OFF"
                    if not GPU_AVAILABLE:
                        gpu_status += " (not available)"
                    cv2.putText(
                        poly_frame,
                        gpu_status,
                        (poly_frame.shape[1] - 350, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    # Display grid points info
                    point_info_text = " + ".join(points_info)
                    cv2.putText(
                        poly_frame,
                        f"Points: {point_info_text}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                    # Display detailed timing information in ms
                    if timing_data:
                        y_pos = 120
                        # Find bottleneck
                        bottleneck = max(
                            timing_data.items(),
                            key=lambda x: x[1] if x[0] != 'total' and x[0] != 'gpu_used' else 0,
                        )
                        bottleneck_name, bottleneck_time = bottleneck

                        cv2.putText(
                            poly_frame,
                            f"Points: {point_count}",
                            (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )
                        y_pos += 30

                        if bottleneck_time > 0.1:  # More than 100ms is slow
                            cv2.putText(
                                poly_frame,
                                f"Bottleneck: {bottleneck_name} ({bottleneck_time*1000:.1f}ms)",
                                (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 255),
                                2,
                            )
                            y_pos += 30

                    if processing_fps < 5:
                        warning_color = (0, 0, 255)  # Red for warning
                        cv2.putText(
                            poly_frame,
                            "Performance Warning: Try reducing point count or enable GPU",
                            (10, poly_frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            warning_color,
                            2,
                        )

                    cv2.putText(
                        poly_frame,
                        f"Freeze: {freeze_mode}",
                        (poly_frame.shape[1] - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                    cv2.imshow('Low-Poly Webcam', poly_frame)

                last_frame = frame.copy()

        finally:
            # Clean up resources
            processing_active = False
            if processing_thread.is_alive():
                processing_thread.join(timeout=1.0)
            cap.release()
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
