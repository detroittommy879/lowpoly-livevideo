# Real-Time Low-Poly Webcam Effect

This project applies a fun, real-time low-polygon (low-poly) visual effect to your webcam feed using Python, OpenCV, and NumPy. It features interactive controls and several optional enhancements for creative experimentation.

## Features

- Real-time low-poly effect on live webcam video
- Adjustable polygon density and quality
- Optional enhancements: edge drawing, grid points, boundary points, color blending, and more
- Interactive parameter tuning with OpenCV trackbars
- Freeze/unfreeze frame functionality
- Clean, modular code with useful comments

## Setup

You are strongly encouraged to use a virtual environment for this project.

### Using Python's built-in venv

#### Windows

```cmd
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Using [uv](https://github.com/astral-sh/uv) (cross-platform, fast alternative)

```bash
uv venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

Install dependencies with:

```bash
pip install -r requirements.txt
```

If you are using uv, you can install dependencies faster with:

```bash
uv pip install -r requirements.txt
```

## Usage

Run the main script:

<details>
<summary><b>Windows</b></summary>

```cmd
python poly.py
```
</details>

<details>
<summary><b>macOS / Linux</b></summary>

```bash
python3 poly.py
```
</details>

- A window will open showing the webcam feed with the low-poly effect.
- Press **ESC** to exit.
- Press **f** to freeze/unfreeze the current frame (if enabled).
- Use the trackbars to adjust parameters in real time (if enabled).

## Configuration

You can enable or disable enhancements and adjust parameters by editing the variables at the top of `poly.py`:

- `DRAW_EDGES`: Draw triangle edges
- `USE_GRID_POINTS`: Add grid points with jitter
- `ADD_BOUNDARY_POINTS`: Add boundary/corner points
- `USE_CENTROID_COLOR`: Use centroid color instead of average
- `BLEND_WITH_ORIGINAL`: Blend low-poly with original frame
- `ENABLE_TRACKBARS`: Enable OpenCV trackbars for real-time tuning
- `FREEZE_FRAME_ENABLED`: Allow freezing the current frame

## Controls

- **ESC**: Exit the application
- **f**: Freeze/unfreeze the frame (if enabled)
- **Trackbars**: Adjust effect parameters live

## Notes

- Make sure your webcam is connected and accessible.
- For best results, experiment with different parameter settings.

---

Enjoy experimenting with real-time low-poly webcam art!
