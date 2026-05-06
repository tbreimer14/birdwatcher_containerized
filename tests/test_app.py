import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
import torch.optim as optim

from src.app import initialize_state


@pytest.fixture(autouse=True)
def clear_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


@patch("src.app.TrickWaterbirdsDataset")
def test_sanity_initialize_state(mock_dataset):
    mock_dataset.return_value = MagicMock()
    initialize_state()

    assert "train_dataset" in st.session_state
    assert "test_dataset" in st.session_state
    assert "current_idx" in st.session_state
    assert "model" in st.session_state
    assert "optimizer" in st.session_state

    assert st.session_state.current_idx == 0
    assert isinstance(st.session_state.optimizer, optim.Optimizer)


@patch("src.app.TrickWaterbirdsDataset")
def test_state_persistence(mock_dataset):
    mock_dataset.return_value = MagicMock()

    # Simulate an already-initialized session where the dataset exists
    st.session_state.train_dataset = mock_dataset.return_value
    st.session_state.current_idx = 5

    # Use a real model to avoid empty parameter list errors in optim.Adam
    from src.model import HookedResNet18

    dummy_model = HookedResNet18()
    st.session_state.model = dummy_model

    initialize_state()

    # Verify existing states were not overwritten
    assert st.session_state.current_idx == 5
    assert st.session_state.model is dummy_model
