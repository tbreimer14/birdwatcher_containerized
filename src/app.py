# src/app.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import torch
from torch.utils.data import Subset, ConcatDataset
import torch.optim as optim
from streamlit_drawable_canvas import st_canvas
import numpy as np

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
    if "train_dataset" not in st.session_state:
        with st.spinner("Loading Training Data..."):
            st.session_state.train_dataset = TrickWaterbirdsDataset(
                split="train", data_dir="./data", filter=True
            )
            st.session_state.current_idx = 0

    if "test_dataset" not in st.session_state:
        with st.spinner("Loading Test Data..."):
            base_test_dataset = TrickWaterbirdsDataset(
                split="test", data_dir="./data", return_pil=False, filter=True
            )

            base_val_dataset = TrickWaterbirdsDataset(
                split="validation", data_dir="./data", return_pil=False, filter=True
            )

            # Combine the datasets
            combined_dataset = ConcatDataset([base_val_dataset, base_test_dataset])

            # Create a fixed random subset (e.g., 500 images) from the combined data
            max_samples = 500
            if max_samples < len(combined_dataset):
                indices = torch.randperm(len(combined_dataset))[:max_samples]
                st.session_state.test_dataset = Subset(combined_dataset, indices)
            else:
                st.session_state.test_dataset = combined_dataset

    if "model" not in st.session_state:
        st.session_state.model = HookedResNet18()

    if "optimizer" not in st.session_state:
        st.session_state.optimizer = optim.Adam(
            st.session_state.model.parameters(), lr=0.001
        )

    if "corrections_made" not in st.session_state:
        st.session_state.corrections_made = 0

    if "current_total_loss" not in st.session_state:
        st.session_state.current_total_loss = None

    if "best_loss" not in st.session_state:
        st.session_state.best_loss = float("inf")

    if "baseline_acc" not in st.session_state:
        with st.spinner("Calculating Initial Baseline on Test Set..."):
            st.session_state.baseline_acc = evaluate_model(
                st.session_state.model, st.session_state.test_dataset
            )

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

        st.metric("Baseline Test Accuracy", f"{st.session_state.baseline_acc:.2f}%")

        st.divider()

        # 2. Progress Tracker
        st.write(f"Corrections made: {st.session_state.corrections_made}")

        if st.session_state.best_loss is not None:
            st.metric("Best Total Loss", f"{st.session_state.best_loss:.3f}")

        # 3. Post-Training Evaluation (Triggered by Loss)
        if st.session_state.best_loss is not None and (
            st.session_state.best_loss < 0.7 or st.session_state.corrections_made >= 3
        ):
            st.success("Target loss achieved. Ready for evaluation.")
            if st.button("Evaluate Post-Training"):
                with st.spinner("Loading Best Weights and Calculating Accuracy..."):
                    # Load the best saved weights into the session state model
                    st.session_state.model.load_state_dict(
                        torch.load("best_model_weights.pth")
                    )

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
        elif st.session_state.best_loss is not None:
            st.warning(
                "Total loss is recommended to be below 0.5 to run post-training evaluation."
            )


def main():
    st.set_page_config(layout="wide")
    st.title("BirdWatcher")

    initialize_state()

    dataset = st.session_state.train_dataset
    model = st.session_state.model
    optimizer = st.session_state.optimizer
    idx = st.session_state.current_idx

    tensor_image, raw_pil_image, label = dataset[idx]
    batch_tensor = tensor_image.unsqueeze(0)
    batch_label = torch.tensor([label])

    st.session_state.current_pil_image = raw_pil_image

    true_class_name = "Waterbird" if label == 1 else "Landbird"

    col1, col2 = st.columns(2)

    # --- Column 1: Current Model State ---
    with col1:
        st.header("Current Output")

        cam, pred_class = get_baseline_cam(model, batch_tensor)
        heatmap_overlay = overlay_heatmap(raw_pil_image, cam)

        pred_class_name = "Waterbird" if pred_class == 1 else "Landbird"

        # Convert the PIL image to a NumPy array
        st.image(
            np.array(heatmap_overlay),
            caption=f"Prediction: {pred_class_name} | Truth: {true_class_name}",
        )

        if "last_loss" in st.session_state:
            st.success(st.session_state.last_loss)
            del st.session_state["last_loss"]

    # --- Column 2: Interact & Train ---
    with col2:
        st.header("Paint Bird")

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=25,
            stroke_color="#ff0000",
            background_image=st.session_state.current_pil_image,
            update_streamlit=True,
            height=224,
            width=224,
            drawing_mode="freedraw",
            key=f"canvas_{idx}",
        )

        if st.button("Teach Model", use_container_width=True):
            if canvas_result.image_data is not None:
                with st.spinner(f"Fine-tuning model..."):
                    user_mask = process_canvas_mask(
                        canvas_result.image_data, target_size=(14, 14)
                    )

                    # Loop the training step
                    for _ in range(10):
                        loss_tot, loss_ce, loss_att = train_step(
                            model,
                            optimizer,
                            batch_tensor,
                            batch_label,
                            user_mask,
                            lambda_weight=1.5,
                        )

                    st.session_state.current_total_loss = loss_tot

                    save_msg = ""
                    if loss_tot < st.session_state.best_loss:
                        st.session_state.best_loss = loss_tot
                        torch.save(model.state_dict(), "best_model_weights.pth")
                        save_msg = " | 💾 New Best Model Saved!"

                    st.session_state.last_loss = (
                        f"Model Updated! Final Total Loss: {loss_tot:.3f}{save_msg}"
                    )
                    st.rerun()

    st.divider()

    # Auto-skip button for finding the next image that needs fixing
    if st.button("Find Next Incorrect Image", use_container_width=True):
        with st.spinner("Scanning dataset for errors..."):
            st.session_state.current_idx = find_next_incorrect_idx(
                model, dataset, st.session_state.current_idx
            )
            st.session_state.corrections_made += 1
        st.rerun()

    render_experiment_tracker()


if __name__ == "__main__":
    main()
