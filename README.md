# 3D Matrix Visualizer – Also watch 3b1b

An interactive, real-time **linear algebra visualizer** built in **Rust** using **eframe (egui)** and **nalgebra**.  
Manipulate **3×3 matrices** and instantly see how they transform 3D space through smooth animations and a CAD-style viewport.

A result of studying and vibe co-coding with a fed different LLM:s

---

## ✨ Features

### 🔢 Matrix Interaction
- **Interactive Matrix Input** – Modify matrix cells in real time and watch the space deform instantly.
- **Smooth Animations** – Transitions are interpolated using *smoothstep* to clearly visualize transformation paths.

### 📍 Vector Tools
- **Custom Vector Placement**
  - Manually input coordinates.
  - Hold **`alt`** to “pick up” and place the vector on the **XY-plane** using the mouse.

### 🧭 CAD-Style Viewport
- **Rotation:** Left-click + drag (Yaw / Pitch)
- **Zoom:** Mouse scroll wheel
- **Navigation Cube:**  
  Interactive widget (top-right) to snap the camera to:
  - **XY**
  - **YZ**
  - **XZ** planes

### 📊 Advanced Analysis
- **Determinant Calculation** – Real-time volume of the transformed unit cube.
- **Orientation Tracking**
  - **Purple:** Right-handed system (det > 0)
  - **Red:** Left-handed system (det < 0)
- **Transformation History**
  - Press **`A`** to apply and stack transformations  
    M_total = M_n × … × M_0
    M -1 is the inverse of M_total'
- **Movement visualisations**
  - Flow field on XY-plane
  - Unit sphere with tangent/radial deformation coloring

---

## ⌨️ Hotkeys

| Key | Action |
|-----|--------|
| **P** | Toggle reference origin planes (static gray grid) |
| **V** | Toggle Perspective / Orthographic projection |
| **A** | Apply current matrix to history |
| **Ctrl + Z** | Undo last applied transformation |
| **C** | Clear history and reset to Identity |
| **E** | Draw lines coinciding with eigenvectors |
| **R/O** | Generate Random/Random Orthogonal Matrix |
| **Alt + number key** | Hold to drag vector |
| **Ctrl** | Hold to disable snapping when placing vector |

---

## 🛠 Technical Stack

- **Language:** Rust — https://www.rust-lang.org/
- **GUI Framework:** egui (via eframe) — https://github.com/emilk/egui
- **Linear Algebra:** nalgebra — https://nalgebra.org/
- **Rendering:** Immediate-mode 2D/3D projection using `egui::Painter`

---

## 🚀 Getting Started

### Prerequisites
Install the Rust toolchain from:  
https://rustup.rs/

Download Visual Studio build tools - Desktop development with C++:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Installation & Execution

Clone the repository:

    git clone https://github.com/your-username/matrix-visualizer.git
    cd matrix-visualizer

Run the project:

in repository directory:

    cargo run --release

---

## 📐 Mathematical Concepts

The application visualizes the **linear transformation**:

    T : R³ → R³

defined by a matrix **M**, where each vector is transformed as:

    v' = Mv

---

### 🎨 Visualization Guide

#### Basis Vectors
The colored arrows represent the **columns of the matrix**:

- **Green:** M × [1, 0, 0]ᵀ — Transformed X-axis  
- **Red:** M × [0, 1, 0]ᵀ — Transformed Y-axis  
- **Blue:** M × [0, 0, 1]ᵀ — Transformed Z-axis  

#### Determinant & Volume
- The volume of the unit cube corresponds to:

    |det(M)|

#### Orientation
- If **det(M) < 0**, the transformation includes a **reflection**.
- This flips the orientation of space and is visualized by the cube turning **red**.

---

Built for **students and developers** who want an intuitive, visual understanding of **3D linear algebra**.
