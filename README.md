# BirdWatcher

BirdWatcher is an interactive, human-in-the-loop machine learning tool designed to identify and correct spurious correlations in image classification. 

High-parameter neural networks often learn to rely on environmental backgrounds rather than actual subject features (e.g., identifying a waterbird purely because of a water background). BirdWatcher addresses this by visualizing the model's spatial activations using Grad-CAM heatmaps. Users can diagnose incorrect focus areas and use a drawing canvas to mask the true subject. The system then utilizes a custom Attention Penalty Loss and freezes specific network layers to maintain performance on CPU during a rapid fine-tuning loop, successfully forcing the model to shift its attention to the correct morphology.

## Running the Project

**1. Install Dependencies**
Ensure you have Python installed, then install the required packages:
```bash
pip install -r requirements.txt
```

**2. Launch the Application**
Start the Streamlit interface from the root directory:
```bash
streamlit run src/app.py
```
*(Note: On the first run, the application will automatically download and cache the `grodino/waterbirds` dataset from Hugging Face into a local `data/` directory).*

**3. Run Tests (Optional)**
To verify the integrity of the data loaders, model hooks, and utility functions, run the test suite:
```bash
pytest
```

## Project Structure

* **`src/app.py`**: The main Streamlit application, handling the UI layout, experiment tracking, and the interactive drawing canvas.
* **`src/model.py`**: Contains `HookedResNet18`, a modified pre-trained ResNet18 that registers forward and backward hooks to expose spatial activations for Grad-CAM generation.
* **`src/data_loader.py`**: Handles downloading and formatting the Waterbirds dataset, specifically filtering for out-of-distribution instances where the background and subject labels conflict.
* **`src/utils.py`**: The core mathematical logic of the project. This includes the Grad-CAM generator, heatmap overlay functions, canvas-to-tensor masking transformations, and the custom attention-penalized training loop.
* **`tests/`**: A comprehensive `pytest` suite ensuring the mathematical and structural stability of the application state, model freezing, and spatial mapping.
