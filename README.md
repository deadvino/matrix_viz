# 3D Matrix Visualizer - Alsy view 3b1b

En interaktiv realtidsvisualiserare för linjär algebra, byggd i **Rust** med ramverken **eframe (egui)** och **nalgebra**. Applikationen låter dig manipulera 3x3-matriser och se hur de transformerar rymden i realtid genom animationer och interaktiv 3D-grafik.



## ✨ Funktioner

* **Interaktiv Matrisinmatning:** Ändra matrisens värden manuellt och se rymden deformeras direkt.
* **Smidiga Animationer:** Transformationsmatriser interpoleras med *smoothstep* för att tydligt visualisera övergången från startläge till mål transformation.
* **Vektormanipulering:** Placera en anpassad vektor (gul) genom att hålla `Space` och klicka/dra i viewporten, eller mata in koordinater numeriskt.
* **CAD-liknande Navigering:**
    * **Rotera:** Vänsterklicka och dra för att ändra Yaw och Pitch.
    * **Zooma:** Scrolla för att komma närmare origo.
    * **Navigeringskub:** En interaktiv kub i hörnet låter dig klicka på specifika plan (t.ex. XY, YZ) för att snabbt låsa vyn.
* **Analysverktyg:**
    * **Determinant:** Beräknar volymen av den transformerade enhetskuben.
    * **Färgkodad Orientering:** Enhetskuben ändrar färg beroende på om matrisen bevarar orienteringen (högerhänt system) eller speglar den (vänsterhänt system).
    * **Slumpgenerering:** Skapa matriser automatiskt för att utforska olika geometriska former.

---

## ⌨️ Kortkommandon

| Tangent | Funktion |
| :--- | :--- |
| **P** | Visa/dölj originalplanet (referensrutnätet) |
| **V** | Växla mellan perspektivisk och ortografisk vy |
| **A** | "Apply" - Lägg till nuvarande matris i historiken (Multiplikation) |
| **Ctrl + Z** | Ångra senaste steget i historiken |
| **C** | Rensa historik och återställ till identitetsmatrisen |
| **Space** | Håll inne för att flytta den gula vektorn med muspekaren |

---

## 🛠 Teknisk Stack

* **Språk:** [Rust](https://www.rust-lang.org/)
* **GUI-ramverk:** [egui](https://github.com/emilk/egui) (via eframe)
* **Linjär Algebra:** [nalgebra](https://nalgebra.org/)
* **Rendering:** Immediate mode 2D/3D projection på `egui::Painter`.

---

## 🚀 Kom igång

### Förutsättningar
Du behöver ha Rust-verktygskedjan installerad (`cargo`, `rustc`). Om du inte har det, installera via [rustup.rs](https://rustup.rs/).

### Installation & Körning
1. Klona detta repository:
   ```bash
   git clone [https://github.com/ditt-användarnamn/matrix-visualizer.git](https://github.com/ditt-användarnamn/matrix-visualizer.git)
   cd matrix-visualizer
