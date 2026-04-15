# tests/test_app.py
import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
import torch.optim as optim

from src.app import initialize_state

# Clear Streamlit's global state before each test
@pytest.fixture(autouse=True)
def clear_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

@patch("src.app.TrickWaterbirdsDataset")
def test_sanity_initialize_state(mock_dataset):
    """
    Sanity Check: The state initializer should properly instantiate the dataset, 
    model, optimizer, and index tracking if they do not exist.
    """
    mock_dataset.return_value = MagicMock()

    initialize_state()

    assert 'dataset' in st.session_state
    assert 'current_idx' in st.session_state
    assert 'model' in st.session_state
    assert 'optimizer' in st.session_state

    assert st.session_state.current_idx == 0
    assert isinstance(st.session_state.optimizer, optim.Optimizer)

@patch("src.app.TrickWaterbirdsDataset")
def test_state_persistence(mock_dataset):
    """
    Non-Trivial Test 1: Verify that initialize_state does not overwrite 
    existing values, which is critical for preserving fine-tuned weights 
    and progress across Streamlit reruns.
    """
    mock_dataset.return_value = MagicMock()

    # Simulate a user having progressed to the 5th image
    st.session_state.current_idx = 5

    # Simulate a dummy model already being in state
    dummy_model = "Pre-trained Model Exists"
    st.session_state.model = dummy_model

    initialize_state()

    # The initializer must respect the existing state
    assert st.session_state.current_idx == 5
    assert st.session_state.model == "Pre-trained Model Exists"
