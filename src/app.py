# src/app.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import torch
import torch.optim as optim
from streamlit_drawable_canvas import st_canvas

from src.data_loader import TrickWaterbirdsDataset
from src.model import HookedResNet18
from src.utils import (
    generate_grad_cam,
    overlay_heatmap,
    process_canvas_mask,
    train_step,
    evaluate_model,
)


def initialize_state():
    """Initializes and persists the core ML components across Streamlit reruns."""
    if "train_dataset" not in st.session_state:
        st.session_state.train_dataset = TrickWaterbirdsDataset(
            split="train", data_dir="./data"
        )
        st.session_state.current_idx = 0

    if "test_dataset" not in st.session_state:
        st.session_state.test_dataset = TrickWaterbirdsDataset(
            split="test", data_dir="./data"
        )

    if "model" not in st.session_state:
        st.session_state.model = HookedResNet18()

    if "optimizer" not in st.session_state:
        st.session_state.optimizer = optim.Adam(
            st.session_state.model.parameters(), lr=0.001
        )

    if "corrections_made" not in st.session_state:
        st.session_state.corrections_made = 0
    if "baseline_acc" not in st.session_state:
        st.session_state.baseline_acc = None
    if "new_acc" not in st.session_state:
        st.session_state.new_acc = None


def get_baseline_cam(model, image_tensor):
    """Executes a forward and backward pass to extract the attention map."""
    model.eval()
    model.zero_grad()

    output = model(image_tensor)
    pred_class = torch.argmax(output, dim=1).item()

    output[0, pred_class].backward(retain_graph=True)

    cam = generate_grad_cam(model.get_activations(), model.get_gradients())
    return cam, pred_class


def find_next_incorrect_idx(model, dataset, start_idx):
    """Scans the dataset to find the next image the model predicts incorrectly."""
    model.eval()
    idx = (start_idx + 1) % len(dataset)

    with torch.no_grad():
        for _ in range(len(dataset)):
            tensor_image, _, label = dataset[idx]
            output = model(tensor_image.unsqueeze(0))
            pred = torch.argmax(output, dim=1).item()

            if pred != label:
                return idx

            idx = (idx + 1) % len(dataset)

    return start_idx


def render_experiment_tracker():
    """Renders the evaluation metrics in the sidebar."""
    with st.sidebar:
        st.header("Experiment Tracker")

        # 1. Baseline Evaluation
        if st.session_state.baseline_acc is None:
            st.write("Run a baseline evaluation on the test set before training.")
            if st.button("Evaluate Baseline"):
                with st.spinner("Calculating Baseline on Test Set..."):
                    st.session_state.baseline_acc = evaluate_model(
                        st.session_state.model, st.session_state.test_dataset
                    )
                st.rerun()
        else:
            st.metric("Baseline Test Accuracy", f"{st.session_state.baseline_acc:.2f}%")

        st.divider()

        # 2. Progress Tracker
        st.write(f"Corrections made: {st.session_state.corrections_made} / 5")
        st.progress(min(st.session_state.corrections_made / 5.0, 1.0))

        # 3. Post-Training Evaluation
        if st.session_state.corrections_made >= 5:
            if st.button("Evaluate Post-Training"):
                with st.spinner("Calculating New Accuracy on Test Set..."):
                    st.session_state.new_acc = evaluate_model(
                        st.session_state.model, st.session_state.test_dataset
                    )

            if st.session_state.new_acc is not None:
                delta = st.session_state.new_acc - st.session_state.baseline_acc
                st.metric(
                    "New Test Accuracy",
                    f"{st.session_state.new_acc:.2f}%",
                    delta=f"{delta:.2f}%",
                )


def main():
    st.set_page_config(layout="wide")
    st.title("Explanatory Interactive Learning: Spurious Correlation Fixer")

    initialize_state()

    dataset = st.session_state.train_dataset
    model = st.session_state.model
    optimizer = st.session_state.optimizer
    idx = st.session_state.current_idx

    tensor_image, raw_pil_image, label = dataset[idx]
    batch_tensor = tensor_image.unsqueeze(0)
    batch_label = torch.tensor([label])

    true_class_name = "Waterbird" if label == 1 else "Landbird"

    col1, col2 = st.columns(2)

    # --- Column 1: Current Model State ---
    with col1:
        st.header("1. Current Model State")

        cam, pred_class = get_baseline_cam(model, batch_tensor)
        heatmap_overlay = overlay_heatmap(raw_pil_image, cam)

        pred_class_name = "Waterbird" if pred_class == 1 else "Landbird"
        st.image(
            heatmap_overlay,
            caption=f"Prediction: {pred_class_name} | Truth: {true_class_name}",
        )

        if "last_loss" in st.session_state:
            st.success(st.session_state.last_loss)
            del st.session_state["last_loss"]

    # --- Column 2: Interact & Train ---
    with col2:
        st.header("2. Guide Model")
        st.write("Paint over the actual bird to penalize background attention.")

        current_lambda = st.slider(
            "Attention Penalty Weight (λ)",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
        )

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=25,  # Thicker stroke for freedraw painting
            stroke_color="#ff0000",
            background_image=raw_pil_image,
            update_streamlit=True,
            height=224,
            width=224,
            drawing_mode="freedraw",  # Switched to freedraw
            key="canvas",
        )

        if st.button("Teach Model", use_container_width=True):
            if canvas_result.image_data is not None:
                with st.spinner("Fine-tuning model..."):
                    # Process mask mapped to layer3's 14x14 spatial resolution
                    user_mask = process_canvas_mask(
                        canvas_result.image_data, target_size=(14, 14)
                    )

                    loss_tot, loss_ce, loss_att = train_step(
                        model,
                        optimizer,
                        batch_tensor,
                        batch_label,
                        user_mask,
                        lambda_weight=current_lambda,
                    )

                    st.session_state.corrections_made += 1
                    st.session_state.last_loss = f"Model Updated! CE Loss: {loss_ce:.3f} | Attn Loss: {loss_att:.3f}"

                    # Rerun to immediately update Column 1 with the new model state
                    st.rerun()

    st.divider()

    # Auto-skip button for finding the next image that needs fixing
    if st.button("Find Next Incorrect Image", use_container_width=True):
        with st.spinner("Scanning dataset for errors..."):
            st.session_state.current_idx = find_next_incorrect_idx(
                model, dataset, st.session_state.current_idx
            )
        st.rerun()

    render_experiment_tracker()


if __name__ == "__main__":
    main()
