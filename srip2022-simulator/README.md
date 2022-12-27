# srip2022-simulator

Code to run and work with the OpenAI / Donkey simulator

## Setup

1. Clone this repository

    ```bash
    git clone https://github.com/UCSD-TriTorch/srip2022-simulator
    ```

2. Clone gym-donkeycar (in a separate folder)

    ```bash
    git clone https://github.com/tawnkramer/gym-donkeycar
    ```

3. Install a virtual environment (either via conda or venv). For example, using venv, you could do the following:

    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

4. Install the necessary dependencies (OpenCV)

    ```bash
    pip install opencv-python

5. Install the gym-donkeycar package (do the installation from the parent folder of gym-donkeycar)

    ```bash
    pip install -e .[gym-donkeycar] # Windows
    pip install -e gym-donkeycar # MacOS
    python3 -m pip install -e .[gym-donkeycar] --user # Linux
    ```

6. Download the latest [Donkeycar Simulator](https://github.com/tawnkramer/gym-donkeycar/releases) for your operating system.

7. Start the simulator

8. Run the collect-images.py script.

    ```bash
    python collect-images.py
    ```
