# dla-fractal-cuda
Diffusion Limited Aggregation algorithm in CUDA and C++

## Running Python Scripts (Benchmarks & Charts)

To run the benchmark and chart generation scripts, you should set up a Python virtual environment and install the required dependencies.

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```
2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the benchmark script (this will generate `resultados.csv`):
   ```bash
   python benchmark.py
   ```
5. Run the charts script (this will generate SVG images in the `imagens/` folder):
   ```bash
   python charts.py
   ```
